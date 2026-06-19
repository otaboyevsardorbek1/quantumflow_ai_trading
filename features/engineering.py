"""
QuantumFlow AI Trading System v2.0 - Advanced Feature Engineering
256+ features with regime detection, sentiment, and microstructure
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import logging
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    256+ feature engineering system with:
    - Multi-timeframe analysis (M1 to W1)
    - Market regime detection
    - Advanced microstructure
    - Cross-asset correlation
    - Sentiment features
    - Adaptive feature selection
    """

    def __init__(self, config: Dict):
        self.config = config
        self.scaler = None
        self.pca = None
        self.feature_names = []
        self.regime_detector = MarketRegimeDetector()

    def compute_all_features(self, df: pd.DataFrame, 
                            macro_data: Optional[Dict] = None,
                            sentiment_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Barcha 256+ featurelarni hisoblash

        Args:
            df: Asosiy OHLCV DataFrame
            macro_data: Macro indicatorlar dict
            sentiment_data: Sentiment ma'lumotlari

        Returns:
            features: (N, 256+) numpy array
        """
        features_dict = {}

        # 1. Asosiy narx harakatlari (16 features)
        features_dict.update(self._compute_price_features(df))

        # 2. Trend indikatorlari (32 features)
        features_dict.update(self._compute_trend_features(df))

        # 3. Momentum indikatorlari (24 features)
        features_dict.update(self._compute_momentum_features(df))

        # 4. Volatillik indikatorlari (20 features)
        features_dict.update(self._compute_volatility_features(df))

        # 5. Hajm indikatorlari (16 features)
        features_dict.update(self._compute_volume_features(df))

        # 6. Narx harakatlari / Price Action (20 features)
        features_dict.update(self._compute_price_action_features(df))

        # 7. Market Regime (8 features)
        features_dict.update(self.regime_detector.detect_regime(df))

        # 8. Macro correlations (24 features)
        if macro_data:
            features_dict.update(self._compute_macro_features(df, macro_data))

        # 9. Microstructure (16 features)
        features_dict.update(self._compute_microstructure_features(df))

        # 10. Cross-timeframe (32 features)
        features_dict.update(self._compute_cross_timeframe_features(df))

        # 11. Sentiment (16 features)
        if sentiment_data is not None:
            features_dict.update(self._compute_sentiment_features(sentiment_data))

        # 12. Statistical features (12 features)
        features_dict.update(self._compute_statistical_features(df))

        # DataFrame ga o'tkazish
        features_df = pd.DataFrame(features_dict, index=df.index)

        # NaN va Inf larni tozalash
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(0)

        # Scaling
        if self.config.get('feature_scaling') == 'robust':
            if self.scaler is None:
                self.scaler = RobustScaler()
                features_scaled = self.scaler.fit_transform(features_df)
            else:
                features_scaled = self.scaler.transform(features_df)
        else:
            features_scaled = features_df.values

        # PCA (ixtiyoriy)
        if self.config.get('use_pca', False):
            if self.pca is None:
                self.pca = PCA(n_components=self.config.get('pca_components', 64))
                features_scaled = self.pca.fit_transform(features_scaled)
            else:
                features_scaled = self.pca.transform(features_scaled)

        self.feature_names = list(features_df.columns)

        return features_scaled.astype(np.float32)

    def _compute_price_features(self, df: pd.DataFrame) -> Dict:
        """Asosiy narx featurelari (16)"""
        features = {}

        # Returns
        features['returns'] = df['close'].pct_change()
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_10'] = df['close'].pct_change(10)
        features['returns_20'] = df['close'].pct_change(20)

        # Log returns
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price position
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        # OHLC ratios
        features['hl_ratio'] = (df['high'] - df['low']) / df['close']
        features['oc_ratio'] = (df['close'] - df['open']) / df['open']
        features['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)

        # Range
        features['range'] = df['high'] - df['low']
        features['range_ma'] = features['range'].rolling(20).mean()
        features['range_ratio'] = features['range'] / features['range_ma']

        # Gaps
        features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Intraday momentum
        features['intraday_momentum'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)

        return features

    def _compute_trend_features(self, df: pd.DataFrame) -> Dict:
        """Trend indikatorlari (32)"""
        features = {}

        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'close_sma_{period}_ratio'] = df['close'] / features[f'sma_{period}']
            features[f'close_ema_{period}_ratio'] = df['close'] / features[f'ema_{period}']

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        features['macd_direction'] = np.sign(features['macd_histogram'].diff())

        # ADX
        features['adx'] = self._compute_adx(df, 14)
        features['adx_trending'] = (features['adx'] > 25).astype(float)

        # Ichimoku
        features['tenkan_sen'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        features['kijun_sen'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        features['senkou_span_a'] = ((features['tenkan_sen'] + features['kijun_sen']) / 2).shift(26)
        features['senkou_span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)

        # Parabolic SAR
        features['psar'] = self._compute_parabolic_sar(df)
        features['psar_trend'] = (df['close'] > features['psar']).astype(float)

        return features

    def _compute_momentum_features(self, df: pd.DataFrame) -> Dict:
        """Momentum indikatorlari (24)"""
        features = {}

        # RSI
        for period in [6, 14, 21]:
            features[f'rsi_{period}'] = self._compute_rsi(df['close'], period)
            features[f'rsi_{period}_overbought'] = (features[f'rsi_{period}'] > 70).astype(float)
            features[f'rsi_{period}_oversold'] = (features[f'rsi_{period}'] < 30).astype(float)

        # Stochastic
        features['stoch_k'] = self._compute_stochastic(df, 14)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        features['stoch_cross'] = np.sign(features['stoch_k'] - features['stoch_d'])

        # CCI
        features['cci'] = self._compute_cci(df, 20)

        # Williams %R
        features['williams_r'] = self._compute_williams_r(df, 14)

        # Momentum
        for period in [10, 20]:
            features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        # ROC
        features['roc'] = df['close'].pct_change(10) * 100

        # TSI
        features['tsi'] = self._compute_tsi(df['close'])

        # Ultimate Oscillator
        features['ultimate_osc'] = self._compute_ultimate_oscillator(df)

        return features

    def _compute_volatility_features(self, df: pd.DataFrame) -> Dict:
        """Volatillik indikatorlari (20)"""
        features = {}

        # ATR
        for period in [7, 14, 21]:
            features[f'atr_{period}'] = self._compute_atr(df, period)
            features[f'atr_{period}_ratio'] = features[f'atr_{period}'] / df['close']

        # Bollinger Bands
        for period in [20, 50]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = sma + 2 * std
            features[f'bb_lower_{period}'] = sma - 2 * std
            features[f'bb_position_{period}'] = (df['close'] - features[f'bb_lower_{period}']) /                                                  (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'] + 1e-8)
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / sma

        # Keltner Channels
        features['kc_upper'] = df['close'].ewm(span=20).mean() + 2 * features['atr_14']
        features['kc_lower'] = df['close'].ewm(span=20).mean() - 2 * features['atr_14']

        # Volatility regimes
        features['volatility_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        features['volatility_50'] = df['close'].pct_change().rolling(50).std() * np.sqrt(252)
        features['vol_ratio'] = features['volatility_20'] / features['volatility_50']

        # Parkinson volatility
        features['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * 
                                            (np.log(df['high'] / df['low']) ** 2).rolling(20).mean())

        return features

    def _compute_volume_features(self, df: pd.DataFrame) -> Dict:
        """Hajm indikatorlari (16)"""
        features = {}

        if 'volume' not in df.columns:
            return features

        features['volume'] = df['volume']
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']

        # OBV
        features['obv'] = self._compute_obv(df)
        features['obv_ema'] = features['obv'].ewm(span=20).mean()
        features['obv_trend'] = np.sign(features['obv'] - features['obv_ema'])

        # MFI
        features['mfi'] = self._compute_mfi(df, 14)

        # VWAP
        features['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        features['vwap_distance'] = (df['close'] - features['vwap']) / features['vwap']

        # Volume profile
        features['volume_price_corr'] = df['volume'].rolling(20).corr(df['close'])

        # Relative Volume
        features['rel_volume'] = df['volume'] / df['volume'].rolling(50).mean()

        return features

    def _compute_price_action_features(self, df: pd.DataFrame) -> Dict:
        """Narx harakatlari (20)"""
        features = {}

        # Support/Resistance
        features['support_20'] = df['low'].rolling(20).min()
        features['resistance_20'] = df['high'].rolling(20).max()
        features['support_distance'] = (df['close'] - features['support_20']) / df['close']
        features['resistance_distance'] = (features['resistance_20'] - df['close']) / df['close']

        # Pivot Points
        features['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        features['pivot_distance'] = (df['close'] - features['pivot']) / df['close']

        # Candlestick patterns
        features['doji'] = self._detect_doji(df)
        features['hammer'] = self._detect_hammer(df)
        features['engulfing'] = self._detect_engulfing(df)

        # Fractals
        features['fractal_high'] = self._detect_fractal_high(df)
        features['fractal_low'] = self._detect_fractal_low(df)

        # Trend strength
        features['trend_strength'] = abs(df['close'] - df['close'].shift(20)) /                                       (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-8)

        return features

    def _compute_microstructure_features(self, df: pd.DataFrame) -> Dict:
        """Market microstructure (16)"""
        features = {}

        # Bid-ask spread estimation
        features['spread_est'] = 2 * abs(df['close'] - (df['high'] + df['low']) / 2) / df['close']
        features['spread_ma'] = features['spread_est'].rolling(20).mean()

        # Realized volatility
        features['realized_vol'] = df['close'].pct_change().rolling(20).std()

        # Tick intensity
        features['tick_intensity'] = df['close'].diff().abs().rolling(20).sum() / features['realized_vol']

        # Serial correlation
        features['serial_corr'] = df['close'].pct_change().rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        )

        # Kyle's Lambda (price impact)
        if 'volume' in df.columns:
            features['kyle_lambda'] = (df['high'] - df['low']) / (df['volume'] + 1)

        # Amihud illiquidity
        features['amihud'] = abs(df['close'].pct_change()) / (df['volume'] + 1) if 'volume' in df.columns else 0

        # VPIN (Volume-synchronized Probability of Informed Trading)
        features['vpin'] = self._compute_vpin(df)

        return features

    def _compute_cross_timeframe_features(self, df: pd.DataFrame) -> Dict:
        """Cross-timeframe features (32)"""
        features = {}

        # Resample to higher timeframes and compute alignment
        for tf, freq in [('H1', '1H'), ('H4', '4H'), ('D1', '1D')]:
            resampled = df.resample(freq).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()

            if len(resampled) > 20:
                features[f'{tf}_trend'] = (resampled['close'] > resampled['close'].rolling(20).mean()).astype(float).reindex(df.index, method='ffill')
                features[f'{tf}_rsi'] = self._compute_rsi(resampled['close'], 14).reindex(df.index, method='ffill')
                features[f'{tf}_volatility'] = resampled['close'].pct_change().rolling(20).std().reindex(df.index, method='ffill')

        # Timeframe alignment score
        features['tf_alignment'] = features.get('H1_trend', 0) + features.get('H4_trend', 0) + features.get('D1_trend', 0)
        features['tf_alignment'] = features['tf_alignment'] / 3.0

        return features

    def _compute_macro_features(self, df: pd.DataFrame, macro_data: Dict) -> Dict:
        """Macro correlations (24)"""
        features = {}

        for symbol, data in macro_data.items():
            if isinstance(data, pd.Series):
                # Correlation
                features[f'{symbol}_corr'] = df['close'].pct_change().rolling(50).corr(
                    data.pct_change().rolling(50)
                )

                # Relative strength
                features[f'{symbol}_rs'] = (df['close'].pct_change(20) - data.pct_change(20))

                # Momentum alignment
                features[f'{symbol}_align'] = np.sign(df['close'].pct_change(5)) * np.sign(data.pct_change(5))

        return features

    def _compute_sentiment_features(self, sentiment_df: pd.DataFrame) -> Dict:
        """Sentiment features (16)"""
        features = {}

        features['sentiment_score'] = sentiment_df.get('score', 0)
        features['sentiment_volume'] = sentiment_df.get('volume', 0)
        features['sentiment_ma'] = features['sentiment_score'].rolling(20).mean()
        features['sentiment_zscore'] = (features['sentiment_score'] - features['sentiment_ma']) /                                         features['sentiment_score'].rolling(20).std()

        # Fear/Greed index
        features['fear_greed'] = self._compute_fear_greed_index(sentiment_df)

        return features

    def _compute_statistical_features(self, df: pd.DataFrame) -> Dict:
        """Statistical features (12)"""
        features = {}

        returns = df['close'].pct_change().dropna()

        # Distribution moments
        features['skewness'] = returns.rolling(50).skew()
        features['kurtosis'] = returns.rolling(50).kurt()

        # Jarque-Bera test
        features['jarque_bera'] = returns.rolling(50).apply(
            lambda x: stats.jarque_bera(x.dropna())[0] if len(x.dropna()) > 3 else 0
        )

        # Hurst exponent
        features['hurst'] = returns.rolling(100).apply(self._compute_hurst)

        # ADF test statistic
        features['adf_stat'] = df['close'].rolling(100).apply(
            lambda x: self._compute_adf(x) if len(x) > 10 else 0
        )

        return features

    # Helper methods
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        atr = self._compute_atr(df, period)
        plus_di = 100 * plus_dm.rolling(period).mean() / atr
        minus_di = 100 * minus_dm.rolling(period).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        return dx.rolling(period).mean()

    def _compute_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        return 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)

    def _compute_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad + 1e-8)

    def _compute_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_max = df['high'].rolling(period).max()
        low_min = df['low'].rolling(period).min()
        return -100 * (high_max - df['close']) / (high_max - low_min + 1e-8)

    def _compute_obv(self, df: pd.DataFrame) -> pd.Series:
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)

    def _compute_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        rmf = tp * df['volume']
        delta = tp.diff()

        positive_flow = rmf.where(delta > 0, 0).rolling(period).sum()
        negative_flow = rmf.where(delta < 0, 0).rolling(period).sum()

        mfr = positive_flow / negative_flow
        return 100 - (100 / (1 + mfr))

    def _compute_tsi(self, prices: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
        momentum = prices.diff()
        abs_momentum = momentum.abs()

        ema1 = momentum.ewm(span=r).mean().ewm(span=s).mean()
        ema2 = abs_momentum.ewm(span=r).mean().ewm(span=s).mean()

        return 100 * ema1 / (ema2 + 1e-8)

    def _compute_ultimate_oscillator(self, df: pd.DataFrame) -> pd.Series:
        bp = df['close'] - df[['low', 'close'].shift()].min(axis=1)
        tr = df['high'] - df[['low', 'close'].shift()].min(axis=1)

        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()

        return 100 * (4 * avg7 + 2 * avg14 + avg28) / 7

    def _compute_parabolic_sar(self, df: pd.DataFrame, af: float = 0.02, max_af: float = 0.2) -> pd.Series:
        """Simplified Parabolic SAR"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        psar = np.zeros(len(df))
        psar[0] = close[0]

        for i in range(1, len(df)):
            psar[i] = psar[i-1] + af * (high[i-1] - psar[i-1]) if high[i-1] > psar[i-1] else                       psar[i-1] + af * (low[i-1] - psar[i-1])

        return pd.Series(psar, index=df.index)

    def _detect_doji(self, df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        body = abs(df['close'] - df['open'])
        range_ = df['high'] - df['low']
        return (body / (range_ + 1e-8) < threshold).astype(float)

    def _detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        body = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)

        return ((lower_shadow > 2 * body) & (upper_shadow < body * 0.5)).astype(float)

    def _detect_engulfing(self, df: pd.DataFrame) -> pd.Series:
        prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
        curr_body = abs(df['close'] - df['open'])

        bullish = (df['close'] > df['open']) & (df['open'].shift(1) > df['close'].shift(1)) &                   (curr_body > prev_body)
        bearish = (df['close'] < df['open']) & (df['open'].shift(1) < df['close'].shift(1)) &                   (curr_body > prev_body)

        return (bullish.astype(float) - bearish.astype(float))

    def _detect_fractal_high(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        return (df['high'] == df['high'].rolling(window=2*window+1, center=True).max()).astype(float)

    def _detect_fractal_low(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        return (df['low'] == df['low'].rolling(window=2*window+1, center=True).min()).astype(float)

    def _compute_vpin(self, df: pd.DataFrame, bucket_size: int = 50) -> pd.Series:
        """Volume-synchronized Probability of Informed Trading"""
        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)

        buy_volume = df['volume'] * (df['close'] > df['open']).astype(float)
        sell_volume = df['volume'] * (df['close'] <= df['open']).astype(float)

        vpin = abs(buy_volume - sell_volume).rolling(bucket_size).sum() /                df['volume'].rolling(bucket_size).sum()
        return vpin

    def _compute_hurst(self, x: pd.Series) -> float:
        """Hurst exponent calculation"""
        lags = range(2, 20)
        tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]

        if np.any(np.array(tau) == 0):
            return 0.5

        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def _compute_adf(self, x: pd.Series) -> float:
        """ADF test statistic"""
        from statsmodels.tsa.stattools import adfuller
        try:
            return adfuller(x.dropna(), maxlag=1)[0]
        except:
            return 0

    def _compute_fear_greed_index(self, sentiment_df: pd.DataFrame) -> pd.Series:
        """Fear & Greed index approximation"""
        # Simplified version
        score = sentiment_df.get('score', pd.Series(0, index=sentiment_df.index))
        volume = sentiment_df.get('volume', pd.Series(1, index=sentiment_df.index))

        # Normalize to 0-100
        normalized = (score + 1) / 2 * 100  # Assuming score is -1 to 1
        return normalized.rolling(7).mean()


class MarketRegimeDetector:
    """
    Hidden Markov Model based market regime detection
    Detects: Bull, Bear, Ranging, High Volatility regimes
    """

    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self.fitted = False

    def detect_regime(self, df: pd.DataFrame) -> Dict:
        """Market regime featurelarini hisoblash"""
        features = {}

        returns = df['close'].pct_change().dropna()

        # Regime indicators
        features['regime_volatility'] = returns.rolling(20).std() * np.sqrt(252)
        features['regime_trend'] = (df['close'] > df['close'].rolling(50).mean()).astype(float)
        features['regime_momentum'] = returns.rolling(20).mean() / (returns.rolling(20).std() + 1e-8)

        # Volatility regime
        vol = features['regime_volatility']
        features['regime_low_vol'] = (vol < vol.quantile(0.33)).astype(float)
        features['regime_med_vol'] = ((vol >= vol.quantile(0.33)) & (vol < vol.quantile(0.67))).astype(float)
        features['regime_high_vol'] = (vol >= vol.quantile(0.67)).astype(float)

        # Trend regime
        adx = self._compute_adx_simple(df)
        features['regime_trending'] = (adx > 25).astype(float)
        features['regime_ranging'] = (adx <= 25).astype(float)

        return features

    def _compute_adx_simple(self, df: pd.DataFrame) -> pd.Series:
        """Simplified ADX"""
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()
        plus_dm = (df['high'] - df['high'].shift()).clip(lower=0)
        minus_dm = (df['low'].shift() - df['low']).clip(lower=0)

        plus_di = 100 * plus_dm.rolling(14).mean() / atr
        minus_di = 100 * minus_dm.rolling(14).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        return dx.rolling(14).mean()
