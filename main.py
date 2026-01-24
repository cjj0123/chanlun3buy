import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import datetime
import sys
import akshare as ak
import time

# --- 策略类保持高内聚 ---
class ChanStrategy:
    def __init__(self, symbol, df):
        self.symbol = symbol
        self.df = df.copy()
        # 预计算指标
        ema12 = self.df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['dif'] = ema12 - ema26
        self.df['dea'] = self.df['dif'].ewm(span=9, adjust=False).mean()
        self.df['macd'] = (self.df['dif'] - self.df['dea']) * 2
        self.df['macd_area'] = self.df['macd'].abs()
        
        self.k_data = pd.DataFrame()
        self.bi = []
        self.zhongshu = None
        self.buy_point = None
        self.process_chan()

    def clean_inclusion(self):
        if len(self.df) < 2: return
        k_list = []
        last_k = {'time': self.df.index[0], 'high': self.df.iloc[0]['High'], 'low': self.df.iloc[0]['Low'], 'idx': 0}
        direction = 1 
        for i in range(1, len(self.df)):
            curr_h, curr_l = self.df.iloc[i]['High'], self.df.iloc[i]['Low']
            if (last_k['high'] >= curr_h and last_k['low'] <= curr_l) or (curr_h >= last_k['high'] and curr_l <= last_k['low']):
                if direction == 1:
                    last_k['high'], last_k['low'] = max(last_k['high'], curr_h), max(last_k['low'], curr_l)
                else:
                    last_k['high'], last_k['low'] = min(last_k['high'], curr_h), min(last_k['low'], curr_l)
            else:
                direction = 1 if curr_h > last_k['high'] else -1
                k_list.append(last_k)
                last_k = {'time': self.df.index[i], 'high': curr_h, 'low': curr_l, 'idx': i}
        k_list.append(last_k)
        self.k_data = pd.DataFrame(k_list)

    def identify_bi(self):
        k = self.k_data
        if len(k) < 5: return
        nodes = []
        for i in range(1, len(k) - 1):
            if k.iloc[i]['high'] > k.iloc[i-1]['high'] and k.iloc[i]['high'] > k.iloc[i+1]['high']:
                nodes.append({'type': 'top', 'val': k.iloc[i]['high'], 'idx': k.iloc[i]['idx'], 'time': k.iloc[i]['time']})
            elif k.iloc[i]['low'] < k.iloc[i-1]['low'] and k.iloc[i]['low'] < k.iloc[i+1]['low']:
                nodes.append({'type': 'bottom', 'val': k.iloc[i]['low'], 'idx': k.iloc[i]['idx'], 'time': k.iloc[i]['time']})
        bi = []
        for n in nodes:
            if not bi: bi.append(n)
            elif n['type'] != bi[-1]['type']:
                if abs(n['idx'] - bi[-1]['idx']) >= 3: bi.append(n)
            else:
                if (n['type'] == 'top' and n['val'] > bi[-1]['val']) or (n['type'] == 'bottom' and n['val'] < bi[-1]['val']): bi[-1] = n
        self.bi = bi

    def analyze_three_buy(self):
        if len(self.bi) < 5: return
        try:
            m1, m2, m3 = self.bi[-5], self.bi[-4], self.bi[-3]
            zg = min(max(m1['val'], m2['val']), max(m2['val'], m3['val']))
            zd = max(min(m1['val'], m2['val']), min(m2['val'], m3['val']))
            if zd >= zg: return
            self.zhongshu = {'zg': zg, 'zd': zd, 'start': m1['time'], 'end': m3['time']}
            b_leave, b_back = self.bi[-2], self.bi[-1]
            if b_back['val'] <= zg: return
            p_leave = self.df.loc[self.bi[-3]['time']:self.bi[-2]['time'], 'macd_area'].sum()
            p_back = self.df.loc[self.bi[-2]['time']:self.bi[-1]['time'], 'macd_area'].sum()
            if p_back > p_leave * 0.8: return 
            if self.df.iloc[-1]['Close'] < b_back['val']: return
            self.buy_point = b_back
        except: pass

    def process_chan(self):
        self.clean_inclusion()
        self.identify_bi()
        self.analyze_three_buy()

    def generate_chart_html(self):
        """生成具有唯一 ID 的图表"""
        if not self.buy_point: return None
        
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.05)
        # K线
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], 
                                     low=self.df['Low'], close=self.df['Close'], name='K线'), row=1, col=1)
        # 笔
        bi_x = [b['time'] for b in self.bi]; bi_y = [b['val'] for b in self.bi]
        fig.add_trace(go.Scatter(x=bi_x, y=bi_y, mode='lines+markers', name='笔', line=dict(color='orange', width=2)), row=1, col=1)
        # 中枢
        if self.zhongshu:
            fig.add_shape(type="rect", x0=self.zhongshu['start'], y0=self.zhongshu['zd'], 
                          x1=self.df.index[-1], y1=self.zhongshu['zg'], 
                          fillcolor="rgba(255,0,0,0.1)", line_width=0, row=1, col=1)
        # MACD
        fig.add_trace(go.Bar(x=self.df.index, y=self.df['macd'], name='MACD'), row=2, col=1)
        
        fig.update_layout(title=f"{self.symbol} 缠论三买分析", xaxis_rangeslider_visible=False, height=600)
        
        # 【核心修正】指定唯一的 div_id 为股票代码，防止浏览器渲染错乱
        safe_id = self.symbol.replace('.', '_').replace('-', '_')
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"chart_{safe_id}")

# --- 辅助函数 ---
def get_tickers(index_name):
    print(f"正在获取 {index_name} 列表...")
    tickers = []
    try:
        if index_name == "HSI":
            df = ak.stock_hk_index_stock_cons_em(symbol="恒生指数")
            tickers = [f"{c[1:] if len(c)==5 else c}.HK" for c in df['代码'].tolist()]
        elif index_name == "HS300":
            df = ak.index_stock_cons(symbol="000300")
            tickers = [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
        elif index_name == "ZZ500":
            df = ak.index_stock_cons(symbol="000905")
            tickers = [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
        elif index_name == "NDX":
            # 硬编码纳指100权重股作为示例
            tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "META", "AMD", "NFLX", "AVGO"]
        else:
            tickers = ["0700.HK"]
    except:
        tickers = ["0700.HK"]
    return sorted(list(set(tickers)))

def process_stock(symbol):
    """隔离式下载与处理"""
    try:
        # 使用 Ticker 对象下载，避免 yf.download 的线程竞争问题
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="59d", interval="30m")
        if data.empty or len(data) < 40: return None
        
        # 修正列名大小写
        data.columns = [c.capitalize() for c in data.columns]
        data.index = data.index.tz_localize(None)
        
        cs = ChanStrategy(symbol, data)
        return cs.generate_chart_html()
    except:
        return None

def main():
    index_arg = sys.argv[1] if len(sys.argv) > 1 else "HSI"
    all_tickers = get_tickers(index_arg)
    
    # 结果容器
    results = {}

    print(f"开始并发扫描 {len(all_tickers)} 只个股...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {executor.submit(process_stock, s): s for s in all_tickers}
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                html_snippet = future.result()
                if html_snippet:
                    results[symbol] = html_snippet
                    print(f"找到信号: {symbol}")
            except:
                pass

    # --- 生成最终 HTML ---
    # 头部加载一次 Plotly JS，减少体积并防止重复加载导致的渲染问题
    html_start = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{index_arg} 选股报告</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: sans-serif; background: #f5f5f5; padding: 20px; }}
            .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ text-align: center; color: #333; }}
        </style>
    </head>
    <body>
        <h1>🚀 {index_arg} 缠论三买扫描报告</h1>
        <p style="text-align:center; color: #666;">更新时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_start)
        if not results:
            f.write("<div class='card'><h2 style='text-align:center;'>今日无信号</h2></div>")
        else:
            # 严格按字母顺序写入，确保不重复
            for symbol in sorted(results.keys()):
                f.write(f"<div class='card'>")
                f.write(f"<h2>股票代码: {symbol}</h2>")
                f.write(results[symbol])
                f.write(f"</div>")
        f.write("</body></html>")
    
    print(f"报告生成完毕，共发现 {len(results)} 个信号。")

if __name__ == "__main__":
    main()
