import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import datetime
import sys
import akshare as ak

class ChanStrategy:
    # ... (此处保持之前的 ChanStrategy 类逻辑不变，包含 clean_inclusion, identify_bi, analyze_three_buy)
    def __init__(self, symbol, df):
        self.symbol = symbol
        self.df = df[['High', 'Low', 'Open', 'Close']].copy()
        self.k_data = pd.DataFrame(); self.bi = []; self.zhongshu = None; self.buy_point = None
        self.process_chan()

    def process_chan(self):
        self.clean_inclusion()
        self.identify_bi()
        self.analyze_three_buy()

    def clean_inclusion(self):
        if len(self.df) < 2: return
        data = []; curr_up = self.df.iloc[0]['High']; curr_down = self.df.iloc[0]['Low']; direction = 1
        for i in range(1, len(self.df)):
            high = self.df.iloc[i]['High']; low = self.df.iloc[i]['Low']
            if (curr_up >= high and curr_down <= low) or (high >= curr_up and low <= curr_down):
                if direction == 1: curr_up, curr_down = max(curr_up, high), max(curr_down, low)
                else: curr_up, curr_down = min(curr_up, high), min(curr_down, low)
            else:
                direction = 1 if high > curr_up else -1
                data.append({'up': curr_up, 'down': curr_down, 'time': self.df.index[i-1], 'idx': i-1})
                curr_up, curr_down = high, low
        data.append({'up': curr_up, 'down': curr_down, 'time': self.df.index[-1], 'idx': len(self.df)-1})
        self.k_data = pd.DataFrame(data)

    def identify_bi(self):
        k = self.k_data
        if len(k) < 5: return
        nodes = []
        for i in range(1, len(k) - 1):
            if k.iloc[i]['up'] > k.iloc[i-1]['up'] and k.iloc[i]['up'] > k.iloc[i+1]['up']:
                nodes.append({'type': 'top', 'val': k.iloc[i]['up'], 'time': k.iloc[i]['time'], 'idx': i})
            elif k.iloc[i]['down'] < k.iloc[i-1]['down'] and k.iloc[i]['down'] < k.iloc[i+1]['down']:
                nodes.append({'type': 'bottom', 'val': k.iloc[i]['down'], 'time': k.iloc[i]['time'], 'idx': i})
        bi = []
        for n in nodes:
            if not bi: bi.append(n)
            elif n['type'] != bi[-1]['type']:
                if abs(n['idx'] - bi[-1]['idx']) >= 3: bi.append(n)
            else:
                if (n['type'] == 'top' and n['val'] > bi[-1]['val']) or (n['type'] == 'bottom' and n['val'] < bi[-1]['val']): bi[-1] = n
        self.bi = bi

    def analyze_three_buy(self):
        if len(self.bi) < 7: return
        try:
            m1, m2, m3 = self.bi[-5], self.bi[-4], self.bi[-3]
            zg = min(max(m1['val'], m2['val']), max(m2['val'], m3['val']))
            zd = max(min(m1['val'], m2['val']), min(m2['val'], m3['val']))
            if zd >= zg: return
            self.zhongshu = {'zg': zg, 'zd': zd, 'start': m1['time'], 'end': m3['time']}
            b_leave, b_back = self.bi[-2], self.bi[-1]
            if b_leave['type'] == 'top' and b_leave['val'] > zg and b_back['type'] == 'bottom' and b_back['val'] > zg:
                self.buy_point = b_back
        except: pass

    def generate_plot_html(self):
        if not self.buy_point: return ""
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], low=self.df['Low'], close=self.df['Close'], name=self.symbol))
        bi_x = [b['time'] for b in self.bi]; bi_y = [b['val'] for b in self.bi]
        fig.add_trace(go.Scatter(x=bi_x, y=bi_y, mode='lines+markers', name='笔', line=dict(color='orange', width=2)))
        if self.zhongshu:
            fig.add_shape(type="rect", x0=self.zhongshu['start'], y0=self.zhongshu['zd'], x1=self.df.index[-1], y1=self.zhongshu['zg'], fillcolor="rgba(255, 0, 0, 0.1)", line_width=1, line_color="red")
        fig.add_annotation(x=self.buy_point['time'], y=self.buy_point['val'], text="三买点", showarrow=True, arrowhead=2, arrowcolor="red")
        fig.update_layout(title=f"{self.symbol} 30min三买分析", xaxis_rangeslider_visible=False, height=500)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

def get_tickers(index_name):
    """获取不同指数的成分股 (适配最新版 akshare)"""
    print(f"正在获取 {index_name} 成分股列表...")
    tickers = []
    try:
        if index_name == "SP500":
            # 标普500：使用更稳定的数据源或增加 User-Agent
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            import requests
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            table = pd.read_html(response.text)
            tickers = table[0]['Symbol'].tolist()
            # yfinance 里的标普500点号需处理，如 BRK.B -> BRK-B
            tickers = [t.replace('.', '-') for t in tickers]

        elif index_name == "HS300":
            # 沪深300 最新接口
            df = ak.index_stock_cons(symbol="000300")
            tickers = [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
            
        elif index_name == "ZZ500":
            # 中证500 最新接口
            df = ak.index_stock_cons(symbol="000905")
            tickers = [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
            
        elif index_name == "HSI":
            # 恒生指数主要成分股（维持硬编码或使用接口）
            df = ak.stock_hk_index_spot_em()
            # 筛选恒生指数成分股（简单演示，取市值前30）
            tickers = [f"{code[1:]}.HK" for code in df['代码'].head(30).tolist()]
            
    except Exception as e:
        print(f"获取 {index_name} 列表失败，原因: {e}")
        # 兜底方案
        if index_name == "HSI": tickers = ["0700.HK", "9988.HK", "3690.HK"]
        
    return tickers

def process_stock(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1mo", interval="30m")
        if len(data) < 30: return None
        data.index = data.index.tz_localize(None)
        cs = ChanStrategy(symbol, data)
        return cs.generate_plot_html()
    except: return None

def main():
    # 获取运行参数 (由 GitHub Actions 传入)
    index_arg = sys.argv[1] if len(sys.argv) > 1 else "HSI"
    tickers = get_tickers(index_arg)
    
    print(f"开始扫描 {index_arg}，共 {len(tickers)} 只股票...")
    charts = []
    # 标普500等大数量扫描建议控制并发，防止被Yahoo封禁
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_stock, s) for s in tickers]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: charts.append(res)

    html_header = f"<html><head><meta charset='utf-8'><title>{index_arg} 三买扫描</title></head><body><h1>🚀 {index_arg} 30min三买报告</h1>"
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_header)
        if not charts: f.write("<h2>今日无信号</h2>")
        else:
            for chart in charts: f.write(f"<div>{chart}</div><hr>")
        f.write(f"<p>更新时间: {datetime.datetime.now()}</p></body></html>")

if __name__ == "__main__":
    main()
