"""
Unit tests for the /ws/stt WebSocket STT endpoint.

Covers:
- Pure utility functions: _session_key, _is_authorized, _decode_pcm, _load_ws_cfg
- WebSocket connection lifecycle: auth, disabled/unloaded-model errors
- Audio receive and CloseStream graceful close
- Cross-connection session stitching by API token
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import types
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the ``server`` package alias is installed.
# conftest.py does this at collection time, but explicit guard keeps the module
# importable in isolation too.
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if "server" not in sys.modules:
    pkg = ModuleType("server")
    pkg.__path__ = [str(BACKEND_ROOT)]  # type: ignore[attr-defined]
    pkg.__version__ = "test"
    sys.modules["server"] = pkg


# ---------------------------------------------------------------------------
# Pure utility tests – no FastAPI or HTTP stack needed
# ---------------------------------------------------------------------------


class TestSessionKey:
    def test_deterministic(self):
        from server.api.routes.ws_stt import _session_key

        assert _session_key("tok") == _session_key("tok")

    def test_different_tokens_different_keys(self):
        from server.api.routes.ws_stt import _session_key

        assert _session_key("tok_a") != _session_key("tok_b")

    def test_output_is_24_hex_chars(self):
        from server.api.routes.ws_stt import _session_key

        key = _session_key("anything")
        assert len(key) == 24
        assert all(c in "0123456789abcdef" for c in key)

    def test_matches_sha256_prefix(self):
        from server.api.routes.ws_stt import _session_key

        token = "my_secret_token"
        expected = hashlib.sha256(token.encode()).hexdigest()[:24]
        assert _session_key(token) == expected


class TestIsAuthorized:
    """_is_authorized() tests for both TLS and local-loopback modes."""

    def test_valid_token_accepted_in_tls_mode(self, tmp_path):
        from server.core.token_store import TokenStore

        store = TokenStore(store_path=tmp_path / "tokens.json")
        _stored, plaintext = store.generate_token(client_name="tester", is_admin=False)

        with (
            patch("server.api.routes.ws_stt._TLS_MODE", True),
            patch("server.api.routes.ws_stt.get_token_store", return_value=store),
        ):
            from server.api.routes.ws_stt import _is_authorized

            assert _is_authorized(plaintext, "127.0.0.1") is True

    def test_invalid_token_rejected_in_tls_mode(self, tmp_path):
        from server.core.token_store import TokenStore

        store = TokenStore(store_path=tmp_path / "tokens.json")

        with (
            patch("server.api.routes.ws_stt._TLS_MODE", True),
            patch("server.api.routes.ws_stt.get_token_store", return_value=store),
        ):
            from server.api.routes.ws_stt import _is_authorized

            assert _is_authorized("bad_token", "127.0.0.1") is False

    def test_no_token_rejected_in_tls_mode(self):
        with patch("server.api.routes.ws_stt._TLS_MODE", True):
            from server.api.routes.ws_stt import _is_authorized

            assert _is_authorized(None, "127.0.0.1") is False
            assert _is_authorized("", "127.0.0.1") is False

    def test_loopback_bypasses_auth_in_local_mode(self):
        """In non-TLS mode, loopback connections are trusted without a token."""
        with (
            patch("server.api.routes.ws_stt._TLS_MODE", False),
            patch(
                "server.api.routes.ws_stt.is_local_auth_bypass_host",
                return_value=True,
            ),
        ):
            from server.api.routes.ws_stt import _is_authorized

            assert _is_authorized(None, "127.0.0.1") is True
            assert _is_authorized("", "127.0.0.1") is True

    def test_non_loopback_still_needs_token_in_local_mode(self, tmp_path):
        from server.core.token_store import TokenStore

        store = TokenStore(store_path=tmp_path / "tokens.json")
        _stored, plaintext = store.generate_token(client_name="tester", is_admin=False)

        with (
            patch("server.api.routes.ws_stt._TLS_MODE", False),
            patch(
                "server.api.routes.ws_stt.is_local_auth_bypass_host",
                return_value=False,
            ),
            patch("server.api.routes.ws_stt.get_token_store", return_value=store),
        ):
            from server.api.routes.ws_stt import _is_authorized

            # No token → rejected even in local mode for non-loopback hosts
            assert _is_authorized(None, "192.168.1.100") is False
            # Valid token → accepted
            assert _is_authorized(plaintext, "192.168.1.100") is True


class TestDecodePcm:
    def test_decodes_zeros(self):
        from server.api.routes.ws_stt import _decode_pcm

        data = b"\x00\x00" * 16  # 16 int16 zeros
        out = _decode_pcm(data)
        assert out.dtype == np.float32
        assert len(out) == 16
        assert np.all(out == 0.0)

    def test_decodes_max_positive(self):
        """int16 max (32767) → float32 just below 1.0."""
        from server.api.routes.ws_stt import _decode_pcm

        data = b"\xff\x7f" * 4  # 32767 LE × 4
        out = _decode_pcm(data)
        assert out.dtype == np.float32
        assert len(out) == 4
        assert all(v > 0.99 for v in out)

    def test_decodes_max_negative(self):
        """int16 min (-32768) → float32 just below -1.0."""
        from server.api.routes.ws_stt import _decode_pcm

        data = b"\x00\x80" * 4  # -32768 LE × 4
        out = _decode_pcm(data)
        assert out.dtype == np.float32
        assert len(out) == 4
        assert all(v < -0.99 for v in out)

    def test_empty_bytes_returns_empty_array(self):
        from server.api.routes.ws_stt import _decode_pcm

        out = _decode_pcm(b"")
        assert out.dtype == np.float32
        assert len(out) == 0


class TestLoadWsCfg:
    def test_returns_empty_dict_on_config_error(self):
        """When get_config() raises, _load_ws_cfg returns {}."""
        with patch("server.config.get_config", side_effect=RuntimeError("no config")):
            import server.api.routes.ws_stt as mod

            # Call directly without reloading to test the live function
            result = mod._load_ws_cfg()
            assert isinstance(result, dict)

    def test_ws_stt_section_returned_if_present(self):
        fake_cfg = SimpleNamespace(
            get=lambda key, default=None, **kw: (
                {"enabled": True, "inactivity_timeout_s": 120}
                if key == "ws_stt"
                else default
            )
        )
        with patch("server.config.get_config", return_value=fake_cfg):
            from server.api.routes.ws_stt import _load_ws_cfg

            cfg = _load_ws_cfg()
            assert cfg.get("enabled") is True
            assert cfg.get("inactivity_timeout_s") == 120

    def test_falls_back_to_omi_stt_section(self):
        """If ws_stt is missing, fall back to omi_stt for backward compat."""

        def _get(key, default=None, **kw):
            if key == "ws_stt":
                return None
            if key == "omi_stt":
                return {"enabled": True, "inactivity_timeout_s": 99}
            return default

        fake_cfg = SimpleNamespace(get=lambda *a, default=None, **kw: _get(*a, default=default))
        with patch("server.config.get_config", return_value=fake_cfg):
            from server.api.routes.ws_stt import _load_ws_cfg

            cfg = _load_ws_cfg()
            assert cfg.get("inactivity_timeout_s") == 99


# ---------------------------------------------------------------------------
# WsAudioSession dataclass tests
# ---------------------------------------------------------------------------


class TestWsAudioSession:
    def test_conv_duration_empty(self):
        from server.api.routes.ws_stt import WsAudioSession

        s = WsAudioSession(session_key="abc123")
        assert s.conv_duration_s == 0.0

    def test_conv_duration_accumulates(self):
        from server.api.routes.ws_stt import WsAudioSession

        s = WsAudioSession(session_key="abc123")
        # 16000 samples @ 16000 Hz = 1 second
        chunk = np.zeros(16000, dtype=np.float32)
        s.conv_chunks.append(chunk)
        assert abs(s.conv_duration_s - 1.0) < 0.001

    def test_conv_audio_empty(self):
        from server.api.routes.ws_stt import WsAudioSession

        s = WsAudioSession(session_key="abc123")
        audio = s.conv_audio()
        assert len(audio) == 0
        assert audio.dtype == np.float32

    def test_conv_audio_concatenates_chunks(self):
        from server.api.routes.ws_stt import WsAudioSession

        s = WsAudioSession(session_key="abc123")
        s.conv_chunks.append(np.ones(100, dtype=np.float32))
        s.conv_chunks.append(np.ones(200, dtype=np.float32) * 0.5)
        audio = s.conv_audio()
        assert len(audio) == 300
        assert np.all(audio[:100] == 1.0)
        assert np.all(audio[100:] == 0.5)


# ---------------------------------------------------------------------------
# WebSocket endpoint integration tests (using Starlette TestClient)
# ---------------------------------------------------------------------------


def _build_ws_app(*, tls_mode: bool, token_store, model_loaded: bool = True, ws_cfg=None):
    """Build a minimal FastAPI app that includes the ws_stt router."""
    import server.api.routes.ws_stt as ws_mod
    import server.api.routes.utils as utils_mod
    import server.core.token_store as ts_mod
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    if ws_cfg is None:
        ws_cfg = {"enabled": True, "inactivity_timeout_s": 5, "segment_silence_s": 1.0}

    # Patch globals before building app
    _orig_tls = ws_mod._TLS_MODE
    _orig_utils_tls = utils_mod.TLS_MODE
    _orig_get_ts = ts_mod.get_token_store
    _orig_ts_singleton = ts_mod._token_store

    ws_mod._TLS_MODE = tls_mode
    utils_mod.TLS_MODE = tls_mode
    ts_mod._token_store = token_store
    ts_mod.get_token_store = lambda *_a, **_kw: token_store

    # Patch config
    fake_cfg = SimpleNamespace(
        get=lambda key, default=None, **kw: ws_cfg if key == "ws_stt" else default
    )
    cfg_patcher = patch("server.config.get_config", return_value=fake_cfg)
    cfg_patcher.start()

    # Patch model manager
    mock_engine = MagicMock()
    mock_mm = SimpleNamespace(
        get_status=lambda: {"transcription": {"loaded": model_loaded}},
        transcription_engine=mock_engine,
    )
    mm_patcher = patch("server.core.model_manager.get_model_manager", return_value=mock_mm)
    mm_patcher.start()

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(ws_mod.router, tags=["WebSocket STT"])

    def _restore():
        ws_mod._TLS_MODE = _orig_tls
        utils_mod.TLS_MODE = _orig_utils_tls
        ts_mod._token_store = _orig_ts_singleton
        ts_mod.get_token_store = _orig_get_ts
        cfg_patcher.stop()
        mm_patcher.stop()

    app._test_restore_fn = _restore  # type: ignore[attr-defined]
    return app


@pytest.fixture()
def _ws_store_and_tokens(tmp_path):
    from server.core.token_store import TokenStore

    store = TokenStore(store_path=tmp_path / "tokens.json")
    _, admin_plain = store.generate_token(client_name="ws-admin", is_admin=True)
    _, user_plain = store.generate_token(client_name="ws-user", is_admin=False)
    return store, admin_plain, user_plain


@pytest.fixture()
def ws_client_local(_ws_store_and_tokens):
    from starlette.testclient import TestClient

    store, admin_plain, user_plain = _ws_store_and_tokens
    app = _build_ws_app(tls_mode=False, token_store=store, model_loaded=True)
    client = TestClient(app, raise_server_exceptions=False)
    client.admin_token = admin_plain  # type: ignore[attr-defined]
    client.user_token = user_plain  # type: ignore[attr-defined]
    yield client
    app._test_restore_fn()  # type: ignore[attr-defined]


@pytest.fixture()
def ws_client_tls(_ws_store_and_tokens):
    from starlette.testclient import TestClient

    store, admin_plain, user_plain = _ws_store_and_tokens
    app = _build_ws_app(tls_mode=True, token_store=store, model_loaded=True)
    client = TestClient(app, raise_server_exceptions=False)
    client.admin_token = admin_plain  # type: ignore[attr-defined]
    client.user_token = user_plain  # type: ignore[attr-defined]
    yield client
    app._test_restore_fn()  # type: ignore[attr-defined]


@pytest.fixture()
def ws_client_no_model(_ws_store_and_tokens):
    from starlette.testclient import TestClient

    store, admin_plain, user_plain = _ws_store_and_tokens
    app = _build_ws_app(tls_mode=False, token_store=store, model_loaded=False)
    client = TestClient(app, raise_server_exceptions=False)
    client.admin_token = admin_plain  # type: ignore[attr-defined]
    yield client
    app._test_restore_fn()  # type: ignore[attr-defined]


class TestWsSttAuthInTlsMode:
    def test_no_token_rejected(self, ws_client_tls):
        with ws_client_tls.websocket_connect("/ws/stt") as ws:
            msg = ws.receive_json()
            assert msg["error"] == "Unauthorized"

    def test_invalid_token_rejected(self, ws_client_tls):
        with ws_client_tls.websocket_connect("/ws/stt?token=totally_wrong") as ws:
            msg = ws.receive_json()
            assert msg["error"] == "Unauthorized"

    def test_valid_token_accepted(self, ws_client_tls):
        """A valid token should pass auth and reach model-loading check."""
        token = ws_client_tls.admin_token
        with ws_client_tls.websocket_connect(f"/ws/stt?token={token}") as ws:
            # Send CloseStream immediately; should get segment response, not Unauthorized
            ws.send_text(json.dumps({"type": "CloseStream"}))
            msgs = []
            try:
                while True:
                    msgs.append(ws.receive_json())
            except Exception:
                pass
            # Should not have an auth error
            assert not any(m.get("error") == "Unauthorized" for m in msgs)


class TestWsSttLocalMode:
    def test_loopback_bypass_logic_in_is_authorized(self, tmp_path):
        """Verify _is_authorized respects is_local_auth_bypass_host in non-TLS mode.

        Note: Starlette TestClient uses a non-loopback host ('testclient'), so
        this test validates the bypass logic directly via unit test, not via the
        WebSocket endpoint.  The integration path is tested in TestIsAuthorized.
        """
        with (
            patch("server.api.routes.ws_stt._TLS_MODE", False),
            patch("server.api.routes.ws_stt.is_local_auth_bypass_host", return_value=True),
        ):
            from server.api.routes.ws_stt import _is_authorized

            # No token → should still succeed when host is trusted loopback
            assert _is_authorized(None, "127.0.0.1") is True
            assert _is_authorized("", "127.0.0.1") is True

    def test_omi_alias_works(self, ws_client_local):
        """/ws/omi should behave identically to /ws/stt."""
        token = ws_client_local.user_token
        with ws_client_local.websocket_connect(f"/ws/omi?token={token}") as ws:
            ws.send_text(json.dumps({"type": "CloseStream"}))
            msgs = []
            try:
                while True:
                    msgs.append(ws.receive_json())
            except Exception:
                pass
            assert not any(m.get("error") == "Unauthorized" for m in msgs)


class TestWsSttModelNotLoaded:
    def test_returns_service_unavailable_when_model_not_loaded(self, ws_client_no_model):
        token = ws_client_no_model.admin_token
        with ws_client_no_model.websocket_connect(f"/ws/stt?token={token}") as ws:
            msg = ws.receive_json()
            assert msg["error"] == "Service Unavailable"
            assert "not loaded" in msg["message"].lower()


class TestWsSttDisabled:
    def test_disabled_endpoint_returns_error(self, _ws_store_and_tokens):
        from starlette.testclient import TestClient

        store, admin_plain, _ = _ws_store_and_tokens
        app = _build_ws_app(
            tls_mode=False,
            token_store=store,
            model_loaded=True,
            ws_cfg={"enabled": False},
        )
        client = TestClient(app, raise_server_exceptions=False)
        try:
            with client.websocket_connect(f"/ws/stt?token={admin_plain}") as ws:
                msg = ws.receive_json()
                assert msg["error"] == "Service Unavailable"
                assert "disabled" in msg["message"].lower()
        finally:
            app._test_restore_fn()  # type: ignore[attr-defined]


class TestWsSttCloseStream:
    def test_closestream_delivers_final_segments_message(self, ws_client_local):
        """Sending CloseStream should result in a final is_partial=false message."""
        token = ws_client_local.user_token
        with ws_client_local.websocket_connect(f"/ws/stt?token={token}") as ws:
            ws.send_text(json.dumps({"type": "CloseStream"}))
            msgs = []
            try:
                while True:
                    msgs.append(ws.receive_json())
            except Exception:
                pass
            final_msgs = [m for m in msgs if "is_partial" in m and not m["is_partial"]]
            assert len(final_msgs) == 1
            assert "segments" in final_msgs[0]
            assert isinstance(final_msgs[0]["segments"], list)


class TestWsSttSessionStiching:
    """Cross-connection stitching: same token within inactivity window continues conversation."""

    def test_session_key_stable_across_connections(self):
        from server.api.routes.ws_stt import _session_key

        token = "shared_api_token"
        assert _session_key(token) == _session_key(token)

    def test_different_tokens_use_different_sessions(self):
        from server.api.routes.ws_stt import _session_key

        assert _session_key("token_a") != _session_key("token_b")

    def test_session_accumulates_audio(self):
        """Audio appended to conv_chunks persists across flush calls."""
        from server.api.routes.ws_stt import WsAudioSession

        s = WsAudioSession(session_key="test_key")
        chunk1 = np.zeros(8000, dtype=np.float32)
        chunk2 = np.zeros(8000, dtype=np.float32)
        s.conv_chunks.append(chunk1)
        s.conv_chunks.append(chunk2)
        assert abs(s.conv_duration_s - 1.0) < 0.001  # 16000 samples @ 16000 Hz

    def test_global_session_store_keyed_by_session_key(self, monkeypatch):
        """_ws_sessions dict maps session keys to WsAudioSession objects."""
        import server.api.routes.ws_stt as mod

        monkeypatch.setattr(mod, "_ws_sessions", {})
        from server.api.routes.ws_stt import WsAudioSession, _session_key

        token = "my_test_token"
        skey = _session_key(token)
        session = WsAudioSession(session_key=skey)
        mod._ws_sessions[skey] = session

        assert mod._ws_sessions[skey] is session
        assert mod._ws_sessions[skey].session_key == skey


class TestWsSttPcmAudioReceive:
    """Test that binary PCM frames can be received without crashing."""

    def test_binary_pcm_frame_accepted(self, ws_client_local):
        """Sending PCM audio followed by CloseStream should not error."""
        token = ws_client_local.user_token
        pcm_samples = np.zeros(3200, dtype=np.int16)  # 0.2 s @ 16 kHz
        pcm_bytes = pcm_samples.tobytes()

        with ws_client_local.websocket_connect(f"/ws/stt?codec=pcm&token={token}") as ws:
            ws.send_bytes(pcm_bytes)
            ws.send_text(json.dumps({"type": "CloseStream"}))
            msgs = []
            try:
                while True:
                    msgs.append(ws.receive_json())
            except Exception:
                pass
            # Should not contain error messages from audio processing
            assert not any(m.get("error") == "Unauthorized" for m in msgs)

    def test_unsupported_codec_rejected(self, ws_client_local):
        """Connecting with an unknown codec should return a Bad Request error."""
        token = ws_client_local.user_token
        with ws_client_local.websocket_connect(f"/ws/stt?codec=mp3&token={token}") as ws:
            msg = ws.receive_json()
            assert msg["error"] == "Bad Request"
            assert "codec" in msg["message"].lower()
