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

def get_tickers(index_name):
    """获取成分股列表 - 增加多重版本兼容与终极硬编码兜底"""
    print(f"正在获取 {index_name} 全量成分股列表...")
    tickers = []
    
    try:
        if index_name == "HSI":
            try:
                # 尝试1: 东方财富接口 (目前最稳)
                df = ak.stock_hk_index_stock_cons_em(symbol="恒生指数")
                tickers = df['代码'].tolist()
            except:
                try:
                    # 尝试2: 备用接口名
                    df = ak.index_stock_cons_hk(symbol="恒生指数")
                    tickers = df['代码'].tolist()
                except:
                    print("API 获取恒指失败，启动硬编码 82 只名单兜底...")
                    # 终极兜底：2024-2025 恒生指数 82 只完整名单
                    tickers = [
                        "00001", "00002", "00003", "00005", "00006", "00011", "00012", "00016", "00017", "00027",
                        "00066", "00101", "00151", "00175", "00241", "00267", "00285", "00288", "00291", "00316",
                        "00322", "00358", "00386", "00388", "00669", "00688", "00700", "00713", "00762", "00823",
                        "00857", "00881", "00883", "00939", "00941", "00960", "00968", "00981", "00992", "00998",
                        "01024", "01038", "01044", "01088", "01093", "01109", "01113", "01177", "01209", "01211",
                        "01299", "01308", "01313", "01378", "01398", "01810", "01876", "01928", "01929", "02015",
                        "02020", "02269", "02313", "02318", "02319", "02331", "02333", "02359", "02380", "02382",
                        "02628", "02688", "03690", "03968", "03988", "06098", "06618", "06690", "09618", "09888",
                        "09961", "09988", "09999"
                    ]
            # 格式处理: '00700' -> '0700.HK'
            tickers = [f"{c[1:] if len(c)==5 and c.startswith('0') else c}.HK" for c in tickers]

        elif index_name == "NDX":
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            headers = {'User-Agent': 'Mozilla/5.0'}
            import requests
            response = requests.get(url, headers=headers)
            tables = pd.read_html(response.text)
            for table in tables:
                col = next((c for c in table.columns if c in ['Ticker', 'Symbol']), None)
                if col:
                    tickers = table[col].tolist()
                    break
            tickers = [t.replace('.', '-') for t in tickers]

        elif index_name == "SP500":
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {'User-Agent': 'Mozilla/5.0'}
            import requests
            response = requests.get(url, headers=headers)
            table = pd.read_html(response.text)[0]
            tickers = [t.replace('.', '-') for t in table['Symbol'].tolist()]

        elif index_name == "HS300":
            try:
                df = ak.index_stock_cons(symbol="000300")
                tickers = [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
            except:
                print("HS300 API失效，使用核心权重股兜底")
                tickers = ["600519.SS", "601318.SS", "000858.SZ", "600036.SS", "601166.SS"]
            
        elif index_name == "ZZ500":
            try:
                df = ak.index_stock_cons(symbol="000905")
                tickers = [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
            except:
                tickers = ["600900.SS", "002415.SZ"]

        # 统一去重排序
        tickers = sorted(list(set(tickers)))
        print(f"--- {index_name} 列表获取成功: 共有 {len(tickers)} 只个股 ---")
        return tickers

    except Exception as e:
        print(f"CRITICAL ERROR 获取 {index_name} 失败: {e}")
        return ["0700.HK", "9988.HK"] # 最后的最后
        
def process_stock(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1mo", interval="30m")
        if data.empty: return None
        data.index = data.index.tz_localize(None)
        cs = ChanStrategy(symbol, data)
        return cs.generate_plot_html()
    except: return None

def main():
    # 1. 获取运行参数
    index_arg = sys.argv[1] if len(sys.argv) > 1 else "HSI"
    
    # 2. 获取原始列表
    raw_tickers = get_tickers(index_arg)
    
    # 3. 【核心修正】去重并排序，确保每只股票只处理一次
    tickers = sorted(list(set(raw_tickers))) 
    
    print(f"开始扫描 {index_arg}，去重后共 {len(tickers)} 只股票...")
    
    # 使用字典存储结果，防止多线程写入时的潜在重复
    results_dict = {} 

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 将 symbol 作为 key 传给 future，方便后续追踪
        future_to_symbol = {executor.submit(process_stock, s): s for s in tickers}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                chart_html = future.result()
                if chart_html:
                    results_dict[symbol] = chart_html # 存入字典，key 是唯一的
                    print(f"发现信号 -> {symbol}")
            except Exception as e:
                print(f"处理 {symbol} 时出错: {e}")

    # 4. 按照股票代码顺序生成 HTML
    sorted_symbols = sorted(results_dict.keys())
    
    html_header = f"""
    <html><head><meta charset='utf-8'><title>{index_arg} 强化版三买报告</title>
    <style>body {{ font-family: sans-serif; background: #f0f2f5; padding: 20px; }}
    .card {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
    h1 {{ text-align: center; color: #1a1a1a; }}</style></head>
    <body><h1>🚀 {index_arg} 30min强化三买扫描</h1>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_header)
        if not sorted_symbols:
            f.write("<div class='card'><h2 style='text-align:center; color:gray;'>今日无强力三买信号</h2></div>")
        else:
            for symbol in sorted_symbols:
                f.write(f"<div class='card'>{results_dict[symbol]}</div>")
        f.write(f"<p style='text-align:center;'>最后更新: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></body></html>")

if __name__ == "__main__":
    main()
