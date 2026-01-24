import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import datetime
import sys
import akshare as ak

import pandas as pd
import numpy as np

class ChanStrategy:
    def __init__(self, symbol, df):
        self.symbol = symbol
        # 预计算：MACD用于背驰判断
        self.prepare_indicators(df)
        self.bi = []
        self.zhongshu = None
        self.buy_point = None
        
        # 核心步骤
        self.process_chan()

    def prepare_indicators(self, df):
        """计算缠论所需的辅助指标"""
        df = df.copy()
        # 计算MACD用于力度对比
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['dif'] = ema12 - ema26
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = (df['dif'] - self.df['dea']) * 2
        # 计算MACD绝对值的累计，模拟“面积”
        df['macd_area'] = df['macd'].abs()
        self.df = df

    def clean_inclusion(self):
        """严格K线包含处理：返回包含处理后的新K线序列"""
        k_list = []
        if len(self.df) < 2: return
        
        # 初始K线
        last_k = {
            'time': self.df.index[0],
            'high': self.df.iloc[0]['High'],
            'low': self.df.iloc[0]['Low'],
            'count': 1 # 记录包含了几根K线
        }
        direction = 1 # 默认向上
        
        for i in range(1, len(self.df)):
            curr_h = self.df.iloc[i]['High']
            curr_l = self.df.iloc[i]['Low']
            
            # 判断包含
            if (last_k['high'] >= curr_h and last_k['low'] <= curr_l) or \
               (curr_h >= last_k['high'] and curr_l <= last_k['low']):
                # 包含发生，根据趋势合并
                if direction == 1:
                    last_k['high'] = max(last_k['high'], curr_h)
                    last_k['low'] = max(last_k['low'], curr_l)
                else:
                    last_k['high'] = min(last_k['high'], curr_h)
                    last_k['low'] = min(last_k['low'], curr_l)
                last_k['count'] += 1
            else:
                # 趋势确认切换
                direction = 1 if curr_h > last_k['high'] else -1
                k_list.append(last_k)
                last_k = {'time': self.df.index[i], 'high': curr_h, 'low': curr_l, 'count': 1, 'idx': i}
        
        k_list.append(last_k)
        self.k_data = pd.DataFrame(k_list)

    def identify_bi(self):
        """标准化笔识别：顶底分型之间至少有1根独立K线（即总计至少5根K线）"""
        k = self.k_data
        if len(k) < 5: return
        
        # 1. 识别初步分型
        potential_nodes = []
        for i in range(1, len(k) - 1):
            if k.iloc[i]['high'] > k.iloc[i-1]['high'] and k.iloc[i]['high'] > k.iloc[i+1]['high']:
                potential_nodes.append({'type': 'top', 'val': k.iloc[i]['high'], 'idx': i, 'time': k.iloc[i]['time']})
            elif k.iloc[i]['low'] < k.iloc[i-1]['low'] and k.iloc[i]['low'] < k.iloc[i+1]['low']:
                potential_nodes.append({'type': 'bottom', 'val': k.iloc[i]['low'], 'idx': i, 'time': k.iloc[i]['time']})
        
        # 2. 笔的连接逻辑：严格校验分型间距
        bi = []
        for node in potential_nodes:
            if not bi:
                bi.append(node)
                continue
            
            last = bi[-1]
            # 必须顶底交替
            if node['type'] == last['type']:
                # 同向分型取极值
                if node['type'] == 'top' and node['val'] > last['val']: bi[-1] = node
                elif node['type'] == 'bottom' and node['val'] < last['val']: bi[-1] = node
            else:
                # 核心优化：顶底之间必须至少相隔 3 根包含处理后的K线
                # 满足标准缠论笔定义的最小距离
                if abs(node['idx'] - last['idx']) >= 3:
                    bi.append(node)
        self.bi = bi

    def get_macd_power(self, start_time, end_time):
        """计算某段走势的MACD力度（面积和最高点）"""
        segment = self.df.loc[start_time:end_time]
        if segment.empty: return 0
        return segment['macd_area'].sum()

    def analyze_three_buy(self):
        """
        三买深度优化逻辑：
        1. 寻找合法中枢
        2. 离开段 vs 回调段 力度对比（背驰校验）
        3. 回调段末端确认
        """
        if len(self.bi) < 7: return
        
        try:
            # 定义最近中枢 (由前三笔重叠构成)
            m1, m2, m3 = self.bi[-5], self.bi[-4], self.bi[-3]
            # 中枢高点取两个顶的低值，中枢低点取两个底的高值
            zg = min(m1['val'], m3['val']) if m1['type'] == 'top' else min(m2['val'], self.bi[-6]['val'])
            zd = max(m1['val'], m3['val']) if m1['type'] == 'bottom' else max(m2['val'], self.bi[-6]['val'])
            
            # 简化版中枢有效判定
            zg = min(max(m1['val'], m2['val']), max(m2['val'], m3['val']))
            zd = max(min(m1['val'], m2['val']), min(m2['val'], m3['val']))
            
            if zd >= zg: return
            self.zhongshu = {'zg': zg, 'zd': zd, 'start': m1['time'], 'end': m3['time']}

            # --- 优化判断：三买点构成 ---
            b_leave = self.bi[-2] # 离开笔
            b_back = self.bi[-1]  # 回调笔 (正在形成或已确认)

            # 条件1：回调笔低点不破ZG
            if b_back['val'] <= zg: return
            
            # 条件2：离开段力度 > 回调段力度 (MACD背驰法)
            # 只有当回调的力量明显弱于上涨力量时，三买才可靠
            power_leave = self.get_macd_power(bi[-3]['time'], bi[-2]['time'])
            power_back = self.get_macd_power(bi[-2]['time'], bi[-1]['time'])
            
            # 优化：回调面积必须小于离开面积的 60% (说明跌不动)
            if power_back > power_leave * 0.6: return

            # 条件3：确认回调笔已经结束 (出现底分型确认)
            last_close = self.df.iloc[-1]['Close']
            if last_close < b_back['val']: return # 还在下跌中
            
            # 判定成功
            self.buy_point = b_back
            
        except Exception as e:
            pass

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
