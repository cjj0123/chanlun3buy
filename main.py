import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import datetime
import sys
import akshare as ak

# --- 缠论专家策略类 ---
class ChanStrategy:
    def __init__(self, symbol, df):
        self.symbol = symbol
        self.df = df.copy()
        self.prepare_indicators()
        self.k_data = pd.DataFrame()
        self.bi = []
        self.zhongshu = None
        self.buy_point = None
        self.analysis_report = "" # 存储判断依据
        self.process_chan()

    def prepare_indicators(self):
        # 标准 MACD 计算
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
            
            # 判定依据一：价格位置
            if b_back['val'] <= zg: return
            
            # 判定依据二：背驰力度
            p_leave = self.df.loc[self.bi[-3]['time']:self.bi[-2]['time'], 'macd_area'].sum()
            p_back = self.df.loc[self.bi[-2]['time']:self.bi[-1]['time'], 'macd_area'].sum()
            power_ratio = p_back / p_leave if p_leave > 0 else 1
            
            if power_ratio > 0.85: return # 回调力度过大
            
            # 确认底分型趋势
            curr_close = self.df.iloc[-1]['Close']
            if curr_close < b_back['val']: return
            
            self.buy_point = b_back
            # 生成诊断报告
            self.analysis_report = (
                f"<b>[形态确认]</b> 30分钟级别出现标准上涨中枢，区间 [{zd:.2f} - {zg:.2f}]。<br>"
                f"<b>[买点依据]</b> 股价向上脱离中枢后回踩，最低价 {b_back['val']:.2f} 远高于中枢上沿 {zg:.2f}。<br>"
                f"<b>[力度对比]</b> 回调段MACD能量仅为离开段的 {power_ratio:.1%}，属于典型的空头衰竭，三买信号成立。"
            )
        except: pass

    def process_chan(self):
        self.clean_inclusion()
        self.identify_bi()
        self.analyze_three_buy()

    def generate_chart_html(self):
        if not self.buy_point: return None
        
        # 调整子图比例，增加 MACD 高度
        fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], shared_xaxes=True, vertical_spacing=0.03)
        
        # 1. K线与笔
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], 
                                     low=self.df['Low'], close=self.df['Close'], name='K线',
                                     increasing_line_color='#ef5350', decreasing_line_color='#26a69a'), row=1, col=1)
        bi_x = [b['time'] for b in self.bi]; bi_y = [b['val'] for b in self.bi]
        fig.add_trace(go.Scatter(x=bi_x, y=bi_y, mode='lines+markers', name='缠论笔', 
                                 line=dict(color='#ffa726', width=2.5), marker=dict(size=6)), row=1, col=1)
        
        # 2. 中枢区域
        if self.zhongshu:
            fig.add_shape(type="rect", x0=self.zhongshu['start'], y0=self.zhongshu['zd'], 
                          x1=self.df.index[-1], y1=self.zhongshu['zg'], 
                          fillcolor="rgba(239, 83, 80, 0.12)", line_width=0, row=1, col=1)
        
        # 3. MACD 视觉优化
        colors = ['#ef5350' if val >= 0 else '#26a69a' for val in self.df['macd']]
        fig.add_trace(go.Bar(x=self.df.index, y=self.df['macd'], name='MACD柱', marker_color=colors), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['dif'], name='DIF', line=dict(color='#2196f3', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['dea'], name='DEA', line=dict(color='#ffeb3b', width=1.5)), row=2, col=1)
        
        # 买点标注
        fig.add_annotation(x=self.buy_point['time'], y=self.buy_point['val'], text="三买信号",
                           showarrow=True, arrowhead=2, arrowcolor="red", font=dict(color="red", size=12), row=1, col=1)

        fig.update_layout(template='plotly_white', xaxis_rangeslider_visible=False, height=750, 
                          margin=dict(l=50, r=50, t=50, b=50), showlegend=True,
                          hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        
        safe_id = self.symbol.replace('.', '_').replace('-', '_')
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"chart_{safe_id}")

# --- 辅助函数 ---
def get_tickers(index_name):
    print(f"正在抓取 {index_name} 成分股列表...")
    try:
        if index_name == "HSI":
            df = ak.stock_hk_index_stock_cons_em(symbol="恒生指数")
            return [f"{c[1:] if len(c)==5 else c}.HK" for c in df['代码'].tolist()]
        elif index_name == "HS300":
            df = ak.index_stock_cons(symbol="000300")
            return [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
        elif index_name == "NDX":
            return ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "META", "AMD", "NFLX", "AVGO", "COST", "ADBE"]
        return ["0700.HK"]
    except: return ["0700.HK"]

def process_stock(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="59d", interval="30m")
        if data.empty or len(data) < 50: return None
        data.columns = [c.capitalize() for c in data.columns]
        data.index = data.index.tz_localize(None)
        cs = ChanStrategy(symbol, data)
        if cs.buy_point:
            return {'symbol': symbol, 'html': cs.generate_chart_html(), 'report': cs.analysis_report}
    except: return None
    return None

def main():
    index_arg = sys.argv[1] if len(sys.argv) > 1 else "HSI"
    all_tickers = sorted(list(set(get_tickers(index_arg))))
    
    results = []
    print(f"开始并发扫描 {len(all_tickers)} 只个股...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {executor.submit(process_stock, s): s for s in all_tickers}
        for future in concurrent.futures.as_completed(future_to_symbol):
            res = future.result()
            if res: results.append(res)

    # 按代码排序
    results.sort(key=lambda x: x['symbol'])

    html_start = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>缠论深度扫描报告 - {index_arg}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; margin: 0; padding: 40px; }}
            .report-container {{ max-width: 1100px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 50px; }}
            .card {{ background: white; border-radius: 15px; overflow: hidden; margin-bottom: 50px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }}
            .card-header {{ background: #2c3e50; color: white; padding: 20px 30px; display: flex; justify-content: space-between; align-items: center; }}
            .reason-box {{ padding: 20px 30px; background: #fff9eb; border-left: 5px solid #ffca28; font-size: 0.95em; line-height: 1.6; color: #5d4037; }}
            .chart-wrapper {{ padding: 10px; }}
            .tag {{ background: #ef5350; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <div class="header">
                <h1>📈 {index_arg} 缠论三买深度选股报告</h1>
                <p>扫描级别：30分钟 | 算法：标准笔+中枢背驰过滤 | 更新时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
    """

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_start)
        if not results:
            f.write("<div class='card' style='padding:50px; text-align:center;'><h2>今日暂无高价值三买信号</h2></div>")
        else:
            for item in results:
                f.write(f"""
                <div class="card">
                    <div class="card-header">
                        <span style="font-size:1.4em; font-weight:bold;">{item['symbol']}</span>
                        <span class="tag">三买确认</span>
                    </div>
                    <div class="reason-box">
                        {item['report']}
                    </div>
                    <div class="chart-wrapper">
                        {item['html']}
                    </div>
                </div>
                """)
        f.write("</div></body></html>")
    print(f"报告已生成，包含 {len(results)} 个信号。")

if __name__ == "__main__":
    main()
