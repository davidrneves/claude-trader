"""Tests for Obsidian daily trade log generation."""

from claude_trader.obsidian import ObsidianLogger


class TestWriteDailyLog:
    def test_creates_markdown_file(self, tmp_path):
        logger = ObsidianLogger(vault_path=tmp_path)
        result = logger.write_daily_log(
            equity="10000",
            cash="5000",
            daily_pnl="150",
            positions=[],
            trades=[],
            analyses=[],
        )
        assert result.exists()
        assert result.suffix == ".md"

    def test_frontmatter_fields(self, tmp_path):
        logger = ObsidianLogger(vault_path=tmp_path)
        path = logger.write_daily_log(
            equity="10000",
            cash="5000",
            daily_pnl="150",
            positions=[],
            trades=[],
            analyses=[],
        )
        content = path.read_text()
        assert "type: trade-log" in content
        assert 'equity: "10000"' in content
        assert 'daily_pnl: "150"' in content
        assert "trades_count: 0" in content

    def test_positions_table(self, tmp_path):
        logger = ObsidianLogger(vault_path=tmp_path)
        path = logger.write_daily_log(
            equity="10000",
            cash="5000",
            daily_pnl="150",
            positions=[
                {
                    "symbol": "AAPL",
                    "qty": 10,
                    "avg_entry": "145.00",
                    "current_price": "150.00",
                    "unrealized_pnl": "50.00",
                }
            ],
            trades=[],
            analyses=[],
        )
        content = path.read_text()
        assert "AAPL" in content
        assert "$145.00" in content

    def test_trades_section(self, tmp_path):
        logger = ObsidianLogger(vault_path=tmp_path)
        path = logger.write_daily_log(
            equity="10000",
            cash="5000",
            daily_pnl="150",
            positions=[],
            trades=[
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 10,
                    "price": 150.00,
                    "rationale": "EMA crossover",
                }
            ],
            analyses=[],
        )
        content = path.read_text()
        assert "BUY" in content
        assert "AAPL" in content
        assert "150.00" in content

    def test_analyses_section(self, tmp_path):
        logger = ObsidianLogger(vault_path=tmp_path)
        path = logger.write_daily_log(
            equity="10000",
            cash="5000",
            daily_pnl="0",
            positions=[],
            trades=[],
            analyses=[
                {
                    "symbol": "AAPL",
                    "signal": "buy",
                    "score": 0.65,
                    "agreement": 3,
                    "contrarian": False,
                }
            ],
        )
        content = path.read_text()
        assert "AAPL" in content
        assert "0.650" in content

    def test_creates_vault_directory(self, tmp_path):
        vault = tmp_path / "nested" / "vault"
        ObsidianLogger(vault_path=vault)
        assert vault.exists()
