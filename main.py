import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import datetime
import sys
import akshare as ak

class ChanStrategy:
    def __init__(self, symbol, df):
        self.symbol = symbol
        # 确保数据量足够计算指标
        if len(df) < 40:
            self.k_data = pd.DataFrame()
            return
            
        self.df = df.copy()
        self.k_data = pd.DataFrame()
        self.bi = []
        self.zhongshu = None
        self.buy_point = None
        
        # 1. 计算技术指标 (MACD)
        self.calculate_indicators()
        # 2. 缠论处理
        self.process_chan()

    def calculate_indicators(self):
        """计算 MACD 指标用于力度过滤"""
        ema12 = self.df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['dif'] = ema12 - ema26
        self.df['dea'] = self.df['dif'].ewm(span=9, adjust=False).mean()
        self.df['macd'] = (self.df['dif'] - self.df['dea']) * 2

    def clean_inclusion(self):
        """K线包含处理"""
        df = self.df
        processed_k = []
        last_up, last_down = df.iloc[0]['High'], df.iloc[0]['Low']
        direction = 1 
        for i in range(1, len(df)):
            curr_up, curr_down = df.iloc[i]['High'], df.iloc[i]['Low']
            if (last_up >= curr_up and last_down <= curr_down) or (curr_up >= last_up and curr_down <= last_down):
                if direction == 1:
                    last_up, last_down = max(last_up, curr_up), max(last_down, curr_down)
                else:
                    last_up, last_down = min(last_up, curr_up), min(last_down, curr_down)
            else:
                direction = 1 if curr_up > last_up else -1
                processed_k.append({'up': last_up, 'down': last_down, 'time': df.index[i-1], 'idx': i-1})
                last_up, last_down = curr_up, curr_down
        processed_k.append({'up': last_up, 'down': last_down, 'time': df.index[-1], 'idx': len(df)-1})
        self.k_data = pd.DataFrame(processed_k)

    def identify_bi(self):
        """识别笔"""
        k = self.k_data
        if len(k) < 5: return
        nodes = []
        for i in range(1, len(k) - 1):
            if k.iloc[i]['up'] > k.iloc[i-1]['up'] and k.iloc[i]['up'] > k.iloc[i+1]['up']:
                nodes.append({'type': 'top', 'val': k.iloc[i]['up'], 'time': k.iloc[i]['time'], 'idx': k.iloc[i]['idx']})
            elif k.iloc[i]['down'] < k.iloc[i-1]['down'] and k.iloc[i]['down'] < k.iloc[i+1]['down']:
                nodes.append({'type': 'bottom', 'val': k.iloc[i]['down'], 'time': k.iloc[i]['time'], 'idx': k.iloc[i]['idx']})
        
        bi = []
        for n in nodes:
            if not bi: bi.append(n)
            elif n['type'] != bi[-1]['type'] and abs(n['idx'] - bi[-1]['idx']) >= 3: bi.append(n)
            elif n['type'] == bi[-1]['type']:
                if (n['type'] == 'top' and n['val'] > bi[-1]['val']) or (n['type'] == 'bottom' and n['val'] < bi[-1]['val']): bi[-1] = n
        self.bi = bi

    def analyze_three_buy(self):
        """强化版三买判断逻辑"""
        if len(self.bi) < 7: return
        try:
            # 1. 中枢定义
            m1, m2, m3 = self.bi[-5], self.bi[-4], self.bi[-3]
            zg = min(max(m1['val'], m2['val']), max(m2['val'], m3['val']))
            zd = max(min(m1['val'], m2['val']), min(m2['val'], m3['val']))
            if zd >= zg: return
            self.zhongshu = {'zg': zg, 'zd': zd, 'start': m1['time'], 'end': m3['time']}

            # 2. 离开段强度过滤 (优化点1: 离开幅度必须超过中枢高度的30%)
            b_leave = self.bi[-2]
            zs_height = zg - zd
            leave_height = b_leave['val'] - zg
            if leave_height < zs_height * 0.3: return 

            # 3. 回调段过滤 (优化点2: 回调低点必须高于 ZG 且留有安全余量)
            b_back = self.bi[-1]
            if b_back['val'] <= zg * 1.002: return # 必须高于上沿0.2%，防止假突破

            # 4. MACD 过滤 (优化点3: DIF 必须在 0 轴附近且拒绝死叉或缩量回调)
            # 获取回调底点对应时间点的 MACD 值
            current_macd = self.df.loc[b_back['time']]
            if current_macd['dif'] < -0.05: return # DIF过低说明下杀力度太强，易跌回中枢

            # 5. 底分型确认 (优化点4: K线形态确认向上)
            last_3_klines = self.df.iloc[-3:]
            if not (last_3_klines.iloc[-1]['Low'] > last_3_klines.iloc[-2]['Low'] and 
                    last_3_klines.iloc[-2]['Low'] < last_3_klines.iloc[-3]['Low']):
                # 如果没有底分型，说明还在下跌中，不是买点
                return

            self.buy_point = b_back
        except: pass

    def process_chan(self):
        self.clean_inclusion()
        self.identify_bi()
        self.analyze_three_buy()

    def generate_plot_html(self):
        if not self.buy_point: return ""
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        # K线
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], 
                                     low=self.df['Low'], close=self.df['Close'], name=self.symbol), row=1, col=1)
        # 笔连线
        bi_x = [b['time'] for b in self.bi]; bi_y = [b['val'] for b in self.bi]
        fig.add_trace(go.Scatter(x=bi_x, y=bi_y, mode='lines', name='缠论笔', line=dict(color='orange', width=2)), row=1, col=1)
        # 中枢
        if self.zhongshu:
            fig.add_shape(type="rect", x0=self.zhongshu['start'], y0=self.zhongshu['zd'], x1=self.df.index[-1], y1=self.zhongshu['zg'], 
                          fillcolor="rgba(255, 0, 0, 0.1)", line_width=1, line_color="red", row=1, col=1)
        # MACD
        fig.add_trace(go.Bar(x=self.df.index, y=self.df['macd'], name='MACD柱'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['dif'], name='DIF', line=dict(color='blue')), row=2, col=1)
        
        # 买点标注
        fig.add_annotation(x=self.buy_point['time'], y=self.buy_point['val'], text="三买(已过滤)", showarrow=True, arrowhead=2, arrowcolor="red", row=1, col=1)
        
        fig.update_layout(title=f"{self.symbol} 强化版三买分析 (MACD+力度过滤)", xaxis_rangeslider_visible=False, height=800)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

# --- 选股逻辑与指数获取保持之前的 get_tickers 和 process_stock 不变 ---
def get_tickers(index_name):
    print(f"获取 {index_name} 列表...")
    try:
        if index_name == "HS300":
            df = ak.index_stock_cons(symbol="000300")
            return [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
        elif index_name == "ZZ500":
            df = ak.index_stock_cons(symbol="000905")
            return [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
        elif index_name == "HSI":
            return ["0700.HK", "9988.HK", "3690.HK", "1810.HK", "9888.HK", "9618.HK", "2015.HK", "0981.HK", "1024.HK", "0992.HK"]
        elif index_name == "SP500":
            return ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "GOOG", "AMZN", "META"]
    except: return ["0700.HK"]

def process_stock(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1mo", interval="30m")
        if data.empty: return None
        data.index = data.index.tz_localize(None)
        cs = ChanStrategy(symbol, data)
        return cs.generate_plot_html()
    except: return None

def main():
    index_arg = sys.argv[1] if len(sys.argv) > 1 else "HSI"
    tickers = get_tickers(index_arg)
    charts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_stock, s) for s in tickers]
        for f in concurrent.futures.as_completed(futures):
            res = f.result(); 
            if res: charts.append(res)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='utf-8'></head><body><h1>{index_arg} 强化版三买报告</h1>")
        if not charts: f.write("<h2>今日无强力三买信号</h2>")
        else:
            for c in charts: f.write(f"<div>{c}</div><hr>")
        f.write("</body></html>")

if __name__ == "__main__":
    main()
