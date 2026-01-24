import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import sys
import akshare as ak
import threading
import copy
import time
from datetime import datetime, timedelta  # <--- 关键是这一行

# 全局锁：防止 yfinance 并发下载时的内存交叉污染
download_lock = threading.Lock()
# 进度锁
counter_lock = threading.Lock()
counter = 0

class ChanStrategy:
    def __init__(self, symbol, df_raw):
        self.symbol = symbol
        # 【隔离1】强制深拷贝数据，确保该实例拥有独立数据副本
        self.df = df_raw.copy(deep=True)
        
        self.prepare_indicators()
        self.k_data = pd.DataFrame()
        self.bi = []
        self.zhongshu = None
        self.buy_point = None
        self.analysis_report = ""
        self.process_chan()

    def prepare_indicators(self):
        # 计算MACD
        close = self.df['Close']
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        self.df['dif'] = ema12 - ema26
        self.df['dea'] = self.df['dif'].ewm(span=9, adjust=False).mean()
        self.df['macd'] = (self.df['dif'] - self.df['dea']) * 2
        self.df['macd_area'] = self.df['macd'].abs()

    def clean_inclusion(self):
        if len(self.df) < 5: return
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
                if (n['type'] == 'top' and n['val'] > bi[-1]['val']) or (n['type'] == 'bottom' and n['val'] < bi[-1]['val']):
                    bi[-1] = n
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
                f"<b>[中枢区间]</b> {zd:.2f} - {zg:.2f}<br>"
                f"<b>[三买点位]</b> {b_back['val']:.2f} (不破上沿)<br>"
                f"<b>[背驰对比]</b> 回调力度 {power_ratio:.1%}"
            )
        except: pass

    def process_chan(self):
        self.clean_inclusion()
        self.identify_bi()
        self.analyze_three_buy()

    def generate_chart_html(self):
        """【隔离2】生成带强唯一标识的HTML，防止渲染串位"""
        if not self.buy_point: return None
        
        fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], shared_xaxes=True, vertical_spacing=0.03)
        # K线
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], 
                                     low=self.df['Low'], close=self.df['Close'], name='K线'), row=1, col=1)
        # 笔
        if self.bi:
            bi_x = [b['time'] for b in self.bi]; bi_y = [b['val'] for b in self.bi]
            fig.add_trace(go.Scatter(x=bi_x, y=bi_y, mode='lines+markers', name='笔', line=dict(color='#ffa726', width=2.5)), row=1, col=1)
        # 中枢
        if self.zhongshu:
            fig.add_shape(type="rect", x0=self.zhongshu['start'], y0=self.zhongshu['zd'], 
                          x1=self.df.index[-1], y1=self.zhongshu['zg'], 
                          fillcolor="rgba(239, 83, 80, 0.15)", line_width=0, row=1, col=1)
        # MACD
        colors = ['#ef5350' if val >= 0 else '#26a69a' for val in self.df['macd']]
        fig.add_trace(go.Bar(x=self.df.index, y=self.df['macd'], name='MACD', marker_color=colors), row=2, col=1)
        
        fig.update_layout(title=f"{self.symbol} 缠论分析报告", xaxis_rangeslider_visible=False, 
                          height=650, template='plotly_white', showlegend=False)
        
        # 强制指定 div_id 包含股票代码，彻底杜绝 ID 碰撞
        safe_id = f"chart_{self.symbol.replace('.', '_').replace('-', '_')}"
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id=safe_id)

def get_tickers(index_name):
    """适配全市场指数的稳健获取逻辑"""
    print(f"--- 正在获取 {index_name} 全量成分股列表 ---")
    tickers = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        if index_name == "SP500":
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(requests.get(url, headers=headers, timeout=15).text)
            tickers = [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
            
        elif index_name == "NDX":
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            table = pd.read_html(requests.get(url, headers=headers, timeout=15).text)
            # 尝试寻找包含 Ticker 的列
            for df in table:
                if 'Ticker' in df.columns:
                    tickers = df['Ticker'].tolist()
                    break
            tickers = [t.replace('.', '-') for t in tickers]

        elif index_name == "HSI":
            # 恒生指数 82 只完整硬编码名单
            tickers = [
                "00001", "00002", "00003", "00005", "00006", "00011", "00012", "00016", "00017", "00027",
                "00066", "00101", "00151", "00175", "00241", "00267", "00285", "00288", "00291", "00316",
                "00322", "00358", "00386", "00388", "00669", "00688", "00700", "00713", "00762", "00823",
                "00857", "00881", "00883", "00939", "00941", "00960", "00968", "00981", "00992", "00998",
                "01024", "01038", "01044", "01088", "01093", "01109", "01113", "01177", "01209", "01211",
                "01299", "01308", "01313", "01347", "01378", "01398", "01810", "01876", "01928", "01929",
                "02015", "02020", "02269", "02313", "02318", "02319", "02331", "02333", "02359", "02380",
                "02382", "02628", "02688", "03690", "03968", "03988", "06098", "06618", "06690", "09618",
                "09888", "09961", "09988", "09999"
            ]
            tickers = [f"{c[1:] if len(c)==5 else c}.HK" for c in tickers]

        elif index_name == "HS300":
            df = ak.index_stock_cons(symbol="000300")
            tickers = [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
            
        elif index_name == "ZZ500":
            df = ak.index_stock_cons(symbol="000905")
            tickers = [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]

        tickers = sorted(list(set(tickers)))
        if not tickers: raise ValueError("获取列表为空")
        print(f"--- 列表获取成功: {index_name} 共计 {len(tickers)} 只 ---")
        return tickers

    except Exception as e:
        print(f"获取 {index_name} 失败: {e}，启用最低兜底")
        return ["AAPL", "NVDA", "0700.HK", "600519.SS"]

def process_stock(symbol, total):
    global counter
    try:
        # 【隔离3】下载环节加锁，防止 yfinance 并发冲突导致数据串写
        with download_lock:
            data_raw = yf.download(symbol, period="59d", interval="30m", progress=False, timeout=20)
        
        with counter_lock:
            counter += 1
            if counter % 10 == 0 or counter == total:
                print(f"进度: [{counter}/{total}] 已下载 {symbol}")

        if data_raw.empty or len(data_raw) < 40: return None
        
        # 整理列名
        if isinstance(data_raw.columns, pd.MultiIndex):
            data_raw.columns = data_raw.columns.get_level_values(0)
        
        data_raw.index = data_raw.index.tz_localize(None)
        
        # 实例化策略（内部执行数据隔离）
        cs = ChanStrategy(symbol, data_raw)
        if cs.buy_point:
            return {'symbol': symbol, 'html': cs.generate_chart_html(), 'report': cs.analysis_report}
    except: return None
    return None

def main():
    global counter
    index_arg = sys.argv[1] if len(sys.argv) > 1 else "HSI"
    tickers = get_tickers(index_arg)
    total = len(tickers)
    
    # 结果字典：Key=股票代码，确保每只股票只会出现一次
    final_results = {}

    print(f"开始并发扫描（最大线程 10）...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_stock, s, total): s for s in tickers}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                # 存入字典自动去重
                final_results[res['symbol']] = res

    # 排序
    sorted_keys = sorted(final_results.keys())

    # --- 最终网页组装 ---
    # 获取北京时间 (UTC+8)
    beijing_time = datetime.now() + timedelta(hours=8)
    beijing_time_str = beijing_time.strftime('%Y-%m-%d %H:%M:%S')

    html_start = f"""
    <!DOCTYPE html><html><head><meta charset="utf-8">
    <title>{index_arg} 缠论选股报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family:'PingFang SC',sans-serif; background:#f0f2f5; padding:20px; margin:0; }}
        .container {{ max-width:1100px; margin:0 auto; }}
        .card {{ background:white; border-radius:12px; margin-bottom:40px; box-shadow:0 4px 15px rgba(0,0,0,0.08); overflow:hidden; }}
        .card-title {{ background:#2c3e50; color:white; padding:15px 25px; font-size:1.3em; font-weight:bold; }}
        .card-reason {{ padding:15px 25px; background:#fff9eb; border-bottom:1px solid #eee; color:#5d4037; line-height:1.6; }}
        .stats {{ text-align:center; padding:30px; color:#666; }}
    </style></head><body><div class="container">
    <h1 style="text-align:center;">📈 {index_arg} 缠论三买深度报告</h1>
    <div class="stats">扫描范围: {index_arg} | 样本总量: {total} | 发现买点: {len(final_results)}<br>
    更新时间 (北京时间): {beijing_time_str}</div>
    """

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_start)
        if not final_results:
            f.write("<div class='card' style='padding:50px;text-align:center;'><h2>今日暂无符合三买形态的标的</h2></div>")
        else:
            for symbol in sorted_keys:
                item = final_results[symbol]
                f.write(f"""
                <div class="card">
                    <div class="card-title">{symbol}</div>
                    <div class="card-reason">{item['report']}</div>
                    <div style="padding:10px;">{item['html']}</div>
                </div>
                """)
        f.write("</div></body></html>")
    
    print(f"扫描完毕，共发现 {len(final_results)} 个信号。")

if __name__ == "__main__":
    main()
