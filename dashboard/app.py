"""
QuantumFlow AI Trading System v2.0 - Real-time Dashboard
Interactive monitoring with Plotly Dash
"""
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

class TradingDashboard:
    """
    Real-time trading dashboard with:
    - Equity curve visualization
    - Position monitoring
    - Risk metrics
    - Agent performance
    - Feature importance
    """

    def __init__(self, config: Dict):
        self.config = config
        self.port = config.get('dashboard_port', 8050)
        self.refresh_interval = config.get('dashboard_refresh_interval', 5)

        self.app = dash.Dash(__name__, external_stylesheets=[dash.dcc.themes.BOOTSTRAP])
        self._setup_layout()
        self._setup_callbacks()

        # Data storage
        self.equity_history = []
        self.trade_history = []
        self.metrics_history = []

    def _setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("🚀 QuantumFlow AI Trading Dashboard", style={'textAlign': 'center'}),

            # Status indicators
            html.Div([
                html.Div([
                    html.H3("Status"),
                    html.Div(id='status-indicator', children="🟢 Running")
                ], className='four columns'),

                html.Div([
                    html.H3("Equity"),
                    html.Div(id='equity-display', children="$100,000.00")
                ], className='four columns'),

                html.Div([
                    html.H3("P&L"),
                    html.Div(id='pnl-display', children="+$0.00 (0.00%)")
                ], className='four columns'),
            ], className='row'),

            # Main charts
            html.Div([
                dcc.Graph(id='equity-chart'),
                dcc.Interval(id='interval-component', interval=self.refresh_interval*1000)
            ]),

            # Metrics
            html.Div([
                html.H2("📊 Performance Metrics"),
                html.Div(id='metrics-table')
            ]),

            # Agent info
            html.Div([
                html.H2("🤖 Agent Information"),
                html.Div(id='agent-info')
            ]),

            # Risk metrics
            html.Div([
                html.H2("🛡️ Risk Metrics"),
                html.Div(id='risk-metrics')
            ]),

        ])

    def _setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            [Output('equity-chart', 'figure'),
             Output('equity-display', 'children'),
             Output('pnl-display', 'children'),
             Output('metrics-table', 'children'),
             Output('agent-info', 'children'),
             Output('risk-metrics', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Update equity chart
            fig = self._create_equity_chart()

            # Update displays
            current_equity = self._get_current_equity()
            initial_equity = 100000.0
            pnl = current_equity - initial_equity
            pnl_pct = (pnl / initial_equity) * 100

            equity_display = f"${current_equity:,.2f}"
            pnl_display = f"{'+' if pnl >= 0 else ''}${pnl:,.2f} ({pnl_pct:+.2f}%)"

            # Update metrics
            metrics_table = self._create_metrics_table()

            # Update agent info
            agent_info = self._create_agent_info()

            # Update risk metrics
            risk_metrics = self._create_risk_metrics()

            return fig, equity_display, pnl_display, metrics_table, agent_info, risk_metrics

    def _create_equity_chart(self) -> go.Figure:
        """Create equity curve chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )

        if len(self.equity_history) > 0:
            equity_df = pd.DataFrame(self.equity_history)

            # Equity curve
            fig.add_trace(
                go.Scatter(
                    x=equity_df['time'],
                    y=equity_df['equity'],
                    mode='lines',
                    name='Equity',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )

            # Drawdown
            peak = equity_df['equity'].cummax()
            drawdown = (peak - equity_df['equity']) / peak

            fig.add_trace(
                go.Scatter(
                    x=equity_df['time'],
                    y=drawdown * 100,
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='red', width=1),
                    fill='tozeroy'
                ),
                row=2, col=1
            )

        fig.update_layout(
            title="Equity Curve & Drawdown",
            height=600,
            showlegend=True
        )

        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)

        return fig

    def _create_metrics_table(self) -> html.Table:
        """Create metrics table"""
        metrics = self._get_current_metrics()

        return html.Table([
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
            html.Tbody([
                html.Tr([html.Td("Total Return"), html.Td(f"{metrics.get('total_return', 0):.2%}")]),
                html.Tr([html.Td("Sharpe Ratio"), html.Td(f"{metrics.get('sharpe_ratio', 0):.2f}")]),
                html.Tr([html.Td("Sortino Ratio"), html.Td(f"{metrics.get('sortino_ratio', 0):.2f}")]),
                html.Tr([html.Td("Max Drawdown"), html.Td(f"{metrics.get('max_drawdown', 0):.2%}")]),
                html.Tr([html.Td("Win Rate"), html.Td(f"{metrics.get('win_rate', 0):.2%}")]),
                html.Tr([html.Td("Profit Factor"), html.Td(f"{metrics.get('profit_factor', 0):.2f}")]),
                html.Tr([html.Td("Total Trades"), html.Td(f"{metrics.get('total_trades', 0)}")]),
            ])
        ])

    def _create_agent_info(self) -> html.Div:
        """Create agent information display"""
        return html.Div([
            html.P("🤖 Transformer Policy Network"),
            html.P("📊 Ensemble: 3 agents (Trend, Mean-Reversion, Breakout)"),
            html.P("🧠 Architecture: 256-dim, 8 heads, 4 layers"),
            html.P("⚡ Device: CUDA" if self.config.get('device') == 'cuda' else "⚡ Device: CPU"),
        ])

    def _create_risk_metrics(self) -> html.Div:
        """Create risk metrics display"""
        risk = self._get_risk_metrics()

        return html.Div([
            html.P(f"🛡️ Daily Loss Limit: {risk.get('daily_loss', 0):.2%} / 3.00%"),
            html.P(f"🛡️ Max Drawdown: {risk.get('current_drawdown', 0):.2%} / 15.00%"),
            html.P(f"🛡️ Position Size: {risk.get('position_size', 0):.2%} / 20.00%"),
            html.P(f"🛡️ Consecutive Losses: {risk.get('consecutive_losses', 0)} / 5"),
            html.P(f"🛡️ Status: {'✅ Safe' if risk.get('safe', True) else '⚠️ Caution'}"),
        ])

    def _get_current_equity(self) -> float:
        """Get current equity"""
        if len(self.equity_history) > 0:
            return self.equity_history[-1]['equity']
        return 100000.0

    def _get_current_metrics(self) -> Dict:
        """Get current metrics"""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
        }

    def _get_risk_metrics(self) -> Dict:
        """Get risk metrics"""
        return {
            'daily_loss': 0.0,
            'current_drawdown': 0.0,
            'position_size': 0.0,
            'consecutive_losses': 0,
            'safe': True,
        }

    def update_data(self, equity: float, metrics: Dict, trades: List[Dict]):
        """Update dashboard data"""
        from datetime import datetime
        self.equity_history.append({
            'time': datetime.now(),
            'equity': equity
        })
        self.metrics_history.append(metrics)
        self.trade_history.extend(trades)

    def run(self, debug: bool = False):
        """Run dashboard"""
        logger.info(f"📊 Dashboard running on http://localhost:{self.port}")
        self.app.run_server(debug=debug, port=self.port)
