import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import datetime

class ChanStrategy:
    def __init__(self, symbol, df):
        self.symbol = symbol
        # 预处理数据
        self.df = df[['High', 'Low', 'Open', 'Close']].copy()
        self.k_data = pd.DataFrame()
        self.bi = []
        self.zhongshu = None
        self.buy_point = None
        
        self.process_chan()

    def process_chan(self):
        """执行缠论逻辑：包含处理 -> 划笔 -> 找中枢 -> 判定三买"""
        self.clean_inclusion()
        self.identify_bi()
        self.analyze_three_buy()

    def clean_inclusion(self):
        """K线包含处理"""
        if len(self.df) < 2: return
        data = []
        # 初始化
        curr_up = self.df.iloc[0]['High']
        curr_down = self.df.iloc[0]['Low']
        direction = 1 
        
        for i in range(1, len(self.df)):
            high = self.df.iloc[i]['High']
            low = self.df.iloc[i]['Low']
            # 判断包含
            if (curr_up >= high and curr_down <= low) or (high >= curr_up and low <= curr_down):
                if direction == 1:
                    curr_up, curr_down = max(curr_up, high), max(curr_down, low)
                else:
                    curr_up, curr_down = min(curr_up, high), min(curr_down, low)
            else:
                direction = 1 if high > curr_up else -1
                data.append({'up': curr_up, 'down': curr_down, 'time': self.df.index[i-1], 'idx': i-1})
                curr_up, curr_down = high, low
        data.append({'up': curr_up, 'down': curr_down, 'time': self.df.index[-1], 'idx': len(self.df)-1})
        self.k_data = pd.DataFrame(data)

    def identify_bi(self):
        """识别缠论笔"""
        k = self.k_data
        if len(k) < 5: return
        nodes = []
        for i in range(1, len(k) - 1):
            if k.iloc[i]['up'] > k.iloc[i-1]['up'] and k.iloc[i]['up'] > k.iloc[i+1]['up']:
                nodes.append({'type': 'top', 'val': k.iloc[i]['up'], 'time': k.iloc[i]['time'], 'idx': i})
            elif k.iloc[i]['down'] < k.iloc[i-1]['down'] and k.iloc[i]['down'] < k.iloc[i+1]['down']:
                nodes.append({'type': 'bottom', 'val': k.iloc[i]['down'], 'time': k.iloc[i]['time'], 'idx': i})
        
        # 笔过滤逻辑
        bi = []
        for n in nodes:
            if not bi: bi.append(n)
            elif n['type'] != bi[-1]['type']:
                if abs(n['idx'] - bi[-1]['idx']) >= 3: bi.append(n)
            else:
                if (n['type'] == 'top' and n['val'] > bi[-1]['val']) or (n['type'] == 'bottom' and n['val'] < bi[-1]['val']):
                    bi[-1] = n
        self.bi = bi

    def analyze_three_buy(self):
        """三买逻辑判断"""
        if len(self.bi) < 7: return
        try:
            # 取最近的一个上涨中枢：下-上-下
            # 我们考察 bi[-5], bi[-4], bi[-3] 构成的区间
            m1, m2, m3 = self.bi[-5], self.bi[-4], self.bi[-3]
            zg = min(max(m1['val'], m2['val']), max(m2['val'], m3['val']))
            zd = max(min(m1['val'], m2['val']), min(m2['val'], m3['val']))
            
            if zd >= zg: return # 不构成重叠中枢

            self.zhongshu = {'zg': zg, 'zd': zd, 'start': m1['time'], 'end': m3['time']}
            
            # 离开笔：bi[-2] 向上
            b_leave = self.bi[-2]
            # 回调笔：bi[-1] 向下
            b_back = self.bi[-1]
            
            # 三买核心：离开段新高 > zg，回调段最低点 > zg
            if b_leave['type'] == 'top' and b_leave['val'] > zg:
                if b_back['type'] == 'bottom' and b_back['val'] > zg:
                    self.buy_point = b_back
        except: pass

    def generate_plot_html(self):
        """生成绘图HTML片段"""
        if not self.buy_point: return ""
        
        fig = make_subplots(rows=1, cols=1)
        # 1. K线
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], 
                                     low=self.df['Low'], close=self.df['Close'], name=f'{self.symbol} K线'))
        # 2. 笔连线
        bi_x = [b['time'] for b in self.bi]
        bi_y = [b['val'] for b in self.bi]
        fig.add_trace(go.Scatter(x=bi_x, y=bi_y, mode='lines+markers', name='缠论笔', line=dict(color='orange', width=2)))
        
        # 3. 中枢方框
        if self.zhongshu:
            fig.add_shape(type="rect", x0=self.zhongshu['start'], y0=self.zhongshu['zd'], 
                          x1=self.df.index[-1], y1=self.zhongshu['zg'],
                          fillcolor="rgba(255, 0, 0, 0.1)", line_width=1, line_color="red")
            
        # 4. 买点标注
        fig.add_annotation(x=self.buy_point['time'], y=self.buy_point['val'], text="三买确认点",
                           showarrow=True, arrowhead=2, arrowcolor="red", font=dict(color="red", size=14))

        fig.update_layout(title=f"【三买确认】{self.symbol} - 30分钟级别分析", xaxis_rangeslider_visible=False, height=600)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

def process_stock(symbol):
    try:
        # 获取30分钟级别数据 (至少需要一个月才能看清中枢)
        data = yf.Ticker(symbol).history(period="1mo", interval="30m")
        if len(data) < 30: return None
        data.index = data.index.tz_localize(None) # 移除时区信息防止绘图报错
        
        cs = ChanStrategy(symbol, data)
        return cs.generate_plot_html()
    except: return None

def main():
    hstech_list = [
        "0700.HK", "9988.HK", "3690.HK", "1810.HK", "9888.HK", 
        "9618.HK", "2015.HK", "2382.HK", "0981.HK", "1024.HK",
        "1314.HK", "0241.HK", "0285.HK", "0772.HK", "0992.HK",
        "1347.HK", "1478.HK", "1610.HK", "1797.HK", "1833.HK",
        "2190.HK", "2269.HK", "2318.HK", "2518.HK", "2688.HK",
        "3888.HK", "6060.HK", "6608.HK", "6618.HK", "9626.HK"
    ]
    
    html_header = """
    <html><head><meta charset="utf-8"><title>恒生科技缠论三买报告</title>
    <style>body { font-family: sans-serif; margin: 20px; background: #f4f4f9; }
    .card { background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 30px; padding: 15px; }
    h1 { color: #333; text-align: center; } </style></head>
    <body><h1>🚀 恒生科技30分钟三买扫描报告</h1>
    <p style='text-align:center; color:#666;'>依据：上涨趋势中，股价向上离开中枢后，次级别回调不破中枢上沿(ZG)</p>
    """
    
    charts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_stock, s) for s in hstech_list]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: charts.append(res)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_header)
        if not charts:
            f.write("<div class='card'><h3 style='text-align:center; color:gray;'>今日恒生科技成分股暂无三买信号</h3></div>")
        else:
            for chart in charts:
                f.write(f"<div class='card'>{chart}</div>")
        f.write(f"<p style='text-align:center;'>更新时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p></body></html>")

if __name__ == "__main__":
    main()
