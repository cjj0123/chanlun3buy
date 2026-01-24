import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import datetime
import sys
import akshare as ak
import threading

# 用于多线程计数的锁
lock = threading.Lock()
counter = 0

class ChanStrategy:
    def __init__(self, symbol, df):
        self.symbol = symbol
        self.df = df.copy()
        self.prepare_indicators()
        self.k_data = pd.DataFrame()
        self.bi = []
        self.zhongshu = None
        self.buy_point = None
        self.analysis_report = ""
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
            power_ratio = p_back / p_leave if p_leave > 0 else 1
            if power_ratio > 0.85: return 
            if self.df.iloc[-1]['Close'] < b_back['val']: return
            self.buy_point = b_back
            self.analysis_report = (
                f"<b>[形态确认]</b> 30min上涨中枢区间 [{zd:.2f} - {zg:.2f}]。<br>"
                f"<b>[强度判定]</b> 回踩低点 {b_back['val']:.2f} 守住上沿。回踩力度仅为突破段的 {power_ratio:.1%}，属于缩量回调。"
            )
        except: pass

    def process_chan(self):
        self.clean_inclusion()
        self.identify_bi()
        self.analyze_three_buy()

    def generate_chart_html(self):
        if not self.buy_point: return None
        fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], shared_xaxes=True, vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], 
                                     low=self.df['Low'], close=self.df['Close'], name='K线'), row=1, col=1)
        bi_x = [b['time'] for b in self.bi]; bi_y = [b['val'] for b in self.bi]
        fig.add_trace(go.Scatter(x=bi_x, y=bi_y, mode='lines+markers', name='笔', line=dict(color='#ffa726', width=2)), row=1, col=1)
        if self.zhongshu:
            fig.add_shape(type="rect", x0=self.zhongshu['start'], y0=self.zhongshu['zd'], x1=self.df.index[-1], y1=self.zhongshu['zg'], 
                          fillcolor="rgba(239, 83, 80, 0.1)", line_width=0, row=1, col=1)
        colors = ['#ef5350' if val >= 0 else '#26a69a' for val in self.df['macd']]
        fig.add_trace(go.Bar(x=self.df.index, y=self.df['macd'], name='MACD柱', marker_color=colors), row=2, col=1)
        fig.update_layout(template='plotly_white', xaxis_rangeslider_visible=False, height=700, showlegend=False)
        safe_id = self.symbol.replace('.', '_').replace('-', '_')
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"chart_{safe_id}")

def get_tickers(index_name):
    """极其稳健的代码列表获取"""
    print(f"--- 正在调用 API 获取 {index_name} 成分股 ---")
    tickers = []
    try:
        if index_name == "HSI":
            # 尝试多个接口获取恒生指数
            try:
                df = ak.stock_hk_index_spot_em()
                # 筛选名称包含“恒生指数”的
                hsi_code = df[df['名称'] == '恒生指数']['代码'].iloc[0]
                cons = ak.stock_hk_index_stock_cons_em(symbol="恒生指数")
                tickers = [f"{c[1:] if len(c)==5 else c}.HK" for c in cons['代码'].tolist()]
            except:
                # 强制备选名单 (82只主要成份股)
                tickers = ["0700.HK", "9988.HK", "3690.HK", "1810.HK", "1299.HK", "0005.HK", "0939.HK", "1398.HK", "2318.HK", "3988.HK", "1810.HK", "0388.HK"]
        elif index_name == "HS300":
            df = ak.index_stock_cons(symbol="000300")
            tickers = [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
        elif index_name == "NDX":
            tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "META", "AMD", "NFLX", "AVGO", "COST", "ADBE", "QCOM", "TXN", "INTC"]
        else:
            tickers = ["0700.HK"]
    except Exception as e:
        print(f"获取列表异常: {e}")
        tickers = ["0700.HK", "9988.HK"]
    
    clean_list = sorted(list(set(tickers)))
    print(f"--- 列表准备完毕: 共计 {len(clean_list)} 只个股 ---")
    return clean_list

def process_stock(symbol, total):
    global counter
    try:
        # 下载数据
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="59d", interval="30m")
        
        with lock:
            counter += 1
            if counter % 10 == 0 or counter == total:
                print(f"进度: [{counter}/{total}] 正在处理 {symbol}...")

        if data.empty or len(data) < 40: return None
        data.columns = [c.capitalize() for c in data.columns]
        data.index = data.index.tz_localize(None)
        
        cs = ChanStrategy(symbol, data)
        if cs.buy_point:
            return {'symbol': symbol, 'html': cs.generate_chart_html(), 'report': cs.analysis_report}
    except: return None
    return None

def main():
    global counter
    index_arg = sys.argv[1] if len(sys.argv) > 1 else "HSI"
    all_tickers = get_tickers(index_arg)
    total_count = len(all_tickers)
    
    results = []
    print(f"开始执行多线程并发扫描...")
    
    # 调低并发数，防止被 Yahoo 封 IP 导致数据为空
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_stock, s, total_count): s for s in all_tickers}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: results.append(res)

    results.sort(key=lambda x: x['symbol'])

    # 网页生成
    html_content = f"""
    <!DOCTYPE html><html><head><meta charset="utf-8">
    <title>{index_arg} 缠论选股报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: sans-serif; background: #f0f2f5; padding: 20px; }}
        .card {{ background: white; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); overflow: hidden; }}
        .card-header {{ background: #2c3e50; color: white; padding: 15px 25px; font-weight: bold; font-size: 1.2em; }}
        .reason {{ padding: 15px 25px; background: #fff9eb; color: #5d4037; line-height: 1.6; border-bottom: 1px solid #eee; }}
        .stats {{ text-align: center; color: #666; margin-bottom: 30px; }}
    </style></head>
    <body>
        <h1 style="text-align:center;">🚀 {index_arg} 30min 缠论三买扫描</h1>
        <div class="stats">
            本次扫描范围: {index_arg} | 样本总量: {total_count} 只 | 发现信号: {len(results)} 只 | 扫描时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    """

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        if not results:
            f.write("<div class='card' style='padding:40px; text-align:center;'><h2>今日该指数下未发现符合三买条件的个股</h2></div>")
        else:
            for item in results:
                f.write(f"""
                <div class="card">
                    <div class="card-header">{item['symbol']}</div>
                    <div class="reason">{item['report']}</div>
                    <div style="padding:10px;">{item['html']}</div>
                </div>
                """)
        f.write("</body></html>")
    
    print(f"扫描完毕。成功生成报告，共发现 {len(results)} 个信号。")

if __name__ == "__main__":
    main()
