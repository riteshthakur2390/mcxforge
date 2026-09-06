import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents_code.agent8_dashboard.app import DashboardAlertAgent
from core.bus import Message, Topic


def test_on_rejected_sends_telegram():
    agent = DashboardAlertAgent()
    
    # Mock socketio.emit and self._telegram
    with patch("agents_code.agent8_dashboard.app.socketio.emit") as mock_emit:
        agent._telegram = MagicMock()
        
        # Setup rejected signal payload
        payload = {
            "timestamp": "2026-06-25T15:00:00+05:30",
            "direction": "BUY_PUT",
            "nifty_price": 24063.80,
            "votes": 4,
            "ml_rank_score": 0.39,
            "ml_confidence": 0.32,
            "est_premium": 92.2,
            "sl_premium": 69.2,
            "target_premium": 138.3,
            "strategies_fired": "SuperTrend+RSI|VWAP+EMA|ValueArea",
            "contract_score": 0.39,
            "premium_source": "LIVE",
            "selection_notes": [
                "Live/cache ATM contract premium for dashboard context; no trade plan was created."
            ],
            "option_symbol": "NIFTY26JUN3024050PE",
        }
        
        asyncio.run(agent.on_rejected(Message(topic=Topic.SIGNAL_REJECTED, payload=payload, source="test")))
        
        # Verify socket emission is still done
        mock_emit.assert_called_once()
        assert mock_emit.call_args[0][0] == "signal_rejected"
        
        # Verify _telegram was called
        agent._telegram.assert_called_once()
        sent_text = agent._telegram.call_args[0][0]
        
        # Assert format and details are correct
        assert "⬇ BUY PUT — REJECTED" in sent_text
        assert "15:00:00" in sent_text
        assert "NIFTY26JUN3024050PE" in sent_text
        assert "NIFTY 24063.80" in sent_text
        assert "Votes 4" in sent_text
        assert "Rank 0.39" in sent_text
        assert "Conf 32%" in sent_text
        assert "PREMIUM\n₹92.2" in sent_text
        assert "STOP LOSS\n₹69.2" in sent_text
        assert "TARGET\n₹138.3" in sent_text
        assert "SuperTrend+RSI, VWAP+EMA, ValueArea" in sent_text
        assert "Contract score 0.39 | Source LIVE" in sent_text
        assert "Live/cache ATM contract premium for dashboard context; no trade plan was created." in sent_text
