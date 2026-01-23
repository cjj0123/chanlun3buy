import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import os

class ChanStrategy:
    def __init__(self, symbol, df):
        self.symbol = symbol
        self.df = df[['High', 'Low', 'Open', 'Close', 'Volume']].copy()
        self.k_data = pd.DataFrame()
        self.bi = []
        self.zhongshu = None
        self.buy_point = None
        self.clean_data()
        
    def clean_data(self):
        df = self.df
        if len(df) < 10: return
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
        k = self.k_data
        if len(k) < 7: return []
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
                if (n['type'] == 'top' and n['val'] > bi[-1]['val']) or (n['type'] == 'bottom' and n['val'] < bi[-1]['val']):
                    bi[-1] = n
        self.bi = bi
        return bi

    def analyze(self):
        bi = self.identify_bi()
        if len(bi) < 6: return False
        try:
            m1, m2, m3 = bi[-5], bi[-4], bi[-3]
            zg = min(max(m1['val'], m2['val']), max(m2['val'], m3['val']))
            zd = max(min(m1['val'], m2['val']), min(m2['val'], m3['val']))
            if zd >= zg: return False
            
            self.zhongshu = {'zg': zg, 'zd': zd, 'start': m1['time'], 'end': m3['time']}
            b_leave, b_back = bi[-2], bi[-1]
            
            if b_leave['type'] == 'top' and b_leave['val'] > zg and b_back['type'] == 'bottom' and b_back['val'] > zg:
                self.buy_point = b_back
                return True
        except: pass
        return False

    def plot_chart(self):
        """生成Plotly交互式图表"""
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        # 1. K线图
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], 
                                     low=self.df['Low'], close=self.df['Close'], name='K线'))
        # 2. 绘制笔
        bi_x = [b['time'] for b in self.bi]
        bi_y = [b['val'] for b in self.bi]
        fig.add_trace(go.Scatter(x=bi_x, y=bi_y, mode='lines+markers', name='缠论笔', line=dict(color='orange', width=2)))
        
        # 3. 绘制中枢区间
        if self.zhongshu:
            fig.add_shape(type="rect", x0=self.zhongshu['start'], y0=self.zhongshu['zd'], 
                          x1=self.df.index[-1], y1=self.zhongshu['zg'],
                          fillcolor="LightSalmon", opacity=0.3, layer="below", line_width=0)
            fig.add_annotation(x=self.zhongshu['start'], y=self.zhongshu['zg'], text="30min中枢", showarrow=False, ysift=10)

        # 4. 标注买点
        if self.buy_point:
            fig.add_annotation(x=self.buy_point['time'], y=self.buy_point['val'], text="三买确认点",
                               showarrow=True, arrowhead=2, arrowcolor="red", ax=0, ay=40, font=dict(color="red", size=14))
        
        fig.update_layout(title=f"{self.symbol} 缠论三买分析报告 (判断依据：向上离开中枢后回抽不破ZG)",
                          xaxis_rangeslider_visible=False, height=600, template="plotly_white")
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

def process_stock(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1mo", interval="30m")
        if data.empty: return None
        data.index = data.index.tz_localize(None)
        
        cs = ChanStrategy(symbol, data)
        if cs.analyze():
            return cs.plot_chart()
    except: pass
    return None

def main():
    hstech_list = ["0700.HK", "9988.HK", "3690.HK", "1810.HK", "9888.HK", "9618.HK", "2015.HK", "2382.HK", "0981.HK", "1024.HK", "1314.HK", "0241.HK", "0285.HK", "0772.HK", "0992.HK", "1347.HK", "1478.HK", "1610.HK", "1797.HK", "1833.HK", "2190.HK", "2269.HK", "2318.HK", "2518.HK", "2688.HK", "3888.HK", "6060.HK", "6608.HK", "6618.HK", "9626.HK"]
    
    html_content = ["<html><head><title>缠论选股报告</title></head><body><h1 style='text-align:center;'>恒生科技30分钟三买扫描报告</h1>"]
    found_any = False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_stock, t) for t in hstech_list]
        for f in concurrent.futures.as_completed(futures):
            chart_html = f.result()
            if chart_html:
                html_content.append(chart_html)
                html_content.append("<hr>")
                found_any = True

    if not found_any:
        html_content.append("<h2 style='color:gray; text-align:center;'>今日暂无符合条件的个股</h2>")
    
    html_content.append(f"<p style='text-align:center;'>最后更新时间: {pd.Timestamp.now()}</p></body></html>")
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))

if __name__ == "__main__":
    main()
