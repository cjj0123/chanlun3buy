iimport pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import datetime
import sys
import akshare as ak

# --- 缠论策略类保持严谨逻辑 ---
class ChanStrategy:
    def __init__(self, symbol, df):
        self.symbol = symbol
        self.df = df.copy()
        self.prepare_indicators()
        self.k_data = pd.DataFrame()
        self.bi = []
        self.zhongshu = None
        self.buy_point = None
        self.process_chan()

    def prepare_indicators(self):
        ema12 = self.df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['dif'] = ema12 - ema26
        self.df['dea'] = self.df['dif'].ewm(span=9, adjust=False).mean()
        self.df['macd'] = (self.df['dif'] - self.df['dea']) * 2
        self.df['macd_area'] = self.df['macd'].abs()

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

    def generate_chart(self):
        if not self.buy_point: return None
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], low=self.df['Low'], close=self.df['Close'], name='K线'), row=1, col=1)
        bi_x = [b['time'] for b in self.bi]; bi_y = [b['val'] for b in self.bi]
        fig.add_trace(go.Scatter(x=bi_x, y=bi_y, mode='lines+markers', name='笔', line=dict(color='orange', width=2)), row=1, col=1)
        if self.zhongshu:
            fig.add_shape(type="rect", x0=self.zhongshu['start'], y0=self.zhongshu['zd'], x1=self.df.index[-1], y1=self.zhongshu['zg'], fillcolor="rgba(255,0,0,0.1)", line_width=0, row=1, col=1)
        fig.add_trace(go.Bar(x=self.df.index, y=self.df['macd'], name='MACD'), row=2, col=1)
        fig.update_layout(title=f"{self.symbol} 强化版三买报告", xaxis_rangeslider_visible=False, height=700, margin=dict(t=50, b=50))
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

# --- 核心函数：单线程获取与清理数据 ---
def get_clean_tickers(index_name):
    print(f"正在获取 {index_name} 成分股列表...")
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
            tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOG", "AMZN", "META", "AMD", "AVGO", "COST", "NFLX", "ADBE"]
        else:
            tickers = ["0700.HK"]
    except:
        tickers = ["0700.HK", "9988.HK"]
    
    # 强制去重
    clean_list = sorted(list(set(tickers)))
    print(f"去重后共 {len(clean_list)} 只股票。")
    return clean_list

def process_single_stock(symbol):
    """
    负责单只股票的下载与计算
    """
    try:
        # 使用 59d 规避 Yahoo 60天限制
        data = yf.download(symbol, period="59d", interval="30m", progress=False, timeout=10)
        if data.empty or len(data) < 40: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.index = data.index.tz_localize(None)
        
        cs = ChanStrategy(symbol, data)
        return cs.generate_chart()
    except:
        return None

def main():
    # 1. 初始化
    index_arg = sys.argv[1] if len(sys.argv) > 1 else "HSI"
    tickers = get_clean_tickers(index_arg)
    
    # 2. 存储容器 (字典保证唯一性)
    # Key: 股票代码, Value: 图表HTML
    final_results = {}

    # 3. 并发扫描
    print(f"开始并发扫描，线程数: 10")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 建立映射
        future_to_symbol = {executor.submit(process_single_stock, s): s for s in tickers}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                chart_html = future.result()
                if chart_html:
                    # 只有找到信号才存入字典
                    final_results[symbol] = chart_html
                    print(f">>> 发现信号: {symbol}")
            except Exception as e:
                print(f"分析 {symbol} 失败: {e}")

    # 4. 一次性生成 HTML (在所有扫描任务彻底结束后)
    print(f"扫描结束，共发现 {len(final_results)} 个信号。开始写入文件...")
    
    html_header = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <title>{index_arg} 缠论选股报告</title>
        <style>
            body {{ font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif; background-color: #f5f7fa; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .card {{ background: white; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.08); margin-bottom: 40px; padding: 20px; }}
            h1 {{ text-align: center; color: #2c3e50; margin-bottom: 30px; }}
            .info {{ text-align: center; color: #7f8c8d; margin-bottom: 50px; font-size: 0.9em; }}
            hr {{ border: 0; border-top: 1px solid #eee; margin: 40px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 {index_arg} 30分钟级别三买扫描报告</h1>
            <div class="info">更新时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 策略：强化版缠论三买</div>
    """

    # 按照股票代码排序写入
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_header)
        
        if not final_results:
            f.write("<div class='card'><h2 style='text-align:center; color:#95a5a6;'>今日未发现符合强力三买的标的</h2></div>")
        else:
            sorted_symbols = sorted(final_results.keys())
            for symbol in sorted_symbols:
                # 从字典中取图表，保证唯一
                f.write(f"<div class='card'>{final_results[symbol]}</div>\n")
        
        f.write("""
            <div class="info">风险提示：量化结果仅供参考，不构成投资建议。</div>
        </div>
    </body>
    </html>
    """)
    
    print("写入完成，任务结束。")

if __name__ == "__main__":
    main()
