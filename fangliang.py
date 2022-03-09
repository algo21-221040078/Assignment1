'''
《量化择时系列（2）：如何运用成交额信息进行择时》 
'''
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import datetime


# ------------ 函数定义 ----------------
# 计算 放量指标
def get_fangliang(data, N, M=60):
    '''
    Function
        计算 放量程度指标 
        #放量程度指标 = (当日成交额对数 − 近𝑁日成交额对数均值)⁄近𝑁日成交额对数标准差
    Parameters
        data     [dataframe]  数据（字段['date','amount']）
    Return
        data     [dataframe]  策略表现数据['date','std_lnamount','fangliang']
    '''
    print(data)
    data.loc[:,'mean_lnamount'] = np.log(data.loc[:,'amount']).rolling(window=N).mean()
    data.loc[:,'std_lnamount'] = np.log(data.loc[:,'amount']).rolling(window=N).std()
    data.loc[:,'fangliang'] = (np.log(data.loc[:,'amount']) - data.loc[:,'mean_lnamount']) /data.loc[:,'std_lnamount']
    return data

# 以成交额为标准 为指数成交额排序
def get_amount_rank(index_data, amount_data):
    # stack展开两个数据
    amount_data = amount_data.set_index('date')
    amount_data = amount_data.stack()
    amount_data.name = 'amount'
    amount_data = amount_data.reset_index()  # 去掉所有索引
    amount_data.rename(columns={'level_1':'code'}, inplace=True)

    index_data = index_data.set_index('20100104')
    index_data = index_data.stack()
    index_data.name = 'code'
    index_data = index_data.reset_index()
    index_data.drop(['level_1'],axis=1, inplace=True)
    index_data.rename(columns={'20100104':'date'}, inplace=True)
    index_data = index_data[(index_data['date']>int(20150401))]
    index_data['is_in_index'] = 1

    # 筛选 股指成分股，并按 交易额大小 排序
    amount_data = pd.merge(amount_data, index_data, on=['date','code'], how='right')
    amount_data['rank'] = amount_data.groupby('date')['amount'].rank(method='first',ascending=False)
    return amount_data

# 计算 分化度指标
def get_fenhuadu(data,N):
    '''
    Function
        计算 分化度指标 
        # 成交分化度时序标准分 = （当日成交分化度 − 近𝑁日成交分化度均值）⁄近𝑁日成交分化度标准差
        # 用幂函数拟合 刻画 成交分化度 b：y = A*x^(−b) →→ lny = lnA +(-b)*lnx
            # y：是降序排列的成交额序列；
            # x：从 1 开始到 N 的自然序列，N 为成分股个数；
    Parameters
        data     [dataframe]  数据（字段['date','amount','rank']）
    Return
        data     [dataframe]  策略表现数据['date','standard_fhd','std_s_fhd']
    '''
    data.dropna(inplace=True)
    data.drop(data[data.amount<=0].index,inplace=True)
    data.loc[:,'lny'] = np.log(data.loc[:,'amount'])
    data.loc[:,'lnx'] = np.log(data.loc[:,'rank'])
    
    fenhuadu = data.groupby('date').apply(lambda x:sm.OLS(x['lny'], sm.add_constant(x['lnx'])).fit().params)
    fenhuadu.reset_index(inplace=True)
    data = fenhuadu.loc[:,['date','lnx']]
    data.loc[:,'lnx'] = - data.loc[:,'lnx']
    data.rename(columns={'lnx':'fenhuadu'},inplace=True)
    data['standard_fhd'] = np.divide(np.subtract(data.loc[:,'fenhuadu'], data.loc[:,'fenhuadu'].rolling(N).mean()),
                                            data.loc[:,'fenhuadu'].rolling(N).std() )
    data['std_s_fhd'] = data.loc[:,'standard_fhd'].rolling(N).std()
    return data

def get_factor(data_newfreq,amount_data,N):
    '''
    Function
        计算 成交额择时综合指标 = （时序放量指标 + 成交分化放缩指标）/ √2
    Parameters
        data     [dataframe]  数据（字段['date','amount','rank']）
    Return
        data     [dataframe]  策略表现数据['date','factor','std_factor']
    '''
    data_fangliang = get_fangliang(data_newfreq, N) #[['date_time','date','amount','open','close']] .reset_index()
    data_fenhuadu = get_fenhuadu(amount_data, N)
    data = pd.merge(data_fenhuadu, data_fangliang,on='date')

    data['factor'] = np.add(data['fangliang'], data['standard_fhd']) / np.sqrt(2)
    
    data['std_add'] = np.add(data['std_lnamount'],data['std_s_fhd'] )
    return data

# 生成买卖信号数据
def get_trading_sig(data, s):
    '''
    Parameters
        data [dateframe]   因子数据（字段['factor']）
        s    [int]         因子阈值（此处为方差前的系数）
    Return
        [dateframe]        信号数据（字段[’factor','sig']）
    '''
    # 以±0.5 倍标准差作为开平仓阈值
    # fangliang > S时，为买入信号=1。fangliang <-S，为卖出信号=-1
    data['std_factor'] = data['factor'].rolling(N).std()
    data['avg_factor'] = data['factor'].rolling(N).mean()
    data['S'] = s*data['std_factor']
    data['upper'] = data['avg_factor'] + data['S']
    data['lower'] = data['avg_factor'] - data['S']

    data['pre_factor'] = data['factor'].shift(1).fillna(0)
    print(data[['pre_factor', 'factor', 'upper','lower']])
    data['sig'] = data.apply(lambda x:1 if (x['factor']>x['upper'] and x['pre_factor']<x['upper']) else(
        -1 if (x['factor']<x['lower'] and x['pre_factor']>x['lower']) else 0), axis=1)
    #data['sig'] = data.apply(lambda x:1 if (x['factor']>x['S'] and x['pre_factor']<x['S']) else(
    #    -1 if (x['factor']<-x['S'] and x['pre_factor']>-x['S']) else 0), axis=1)
    
    data.drop(['pre_factor'], axis=1, inplace=True)

    return data

def draw_factor(data,path):
    '''
    Function
        绘制 开盘价 vs 因子,阈值图
    Parameters
        data [dateframe]   因子数据（字段['date_time','factor','open','S']）
    '''
    data.set_index(['date_time'], inplace=True)
    fig,ax = plt.subplots()
    line1 = ax.plot(data['factor'],'b-',label="factor",linewidth=1.0)
    line2 = ax.plot(data['S'],'--',color='orange',label="long",linewidth=1.0)
    line3 = ax.plot(-data['S'],'r--',label="short",linewidth=1.0)
    ax2 = ax.twinx()
    line4 = ax2.plot(data['open'],'k-',label="open",linewidth=1.0)
    
    ax.set_xlabel("date")
    ax.set_ylabel('factor')
    ax2.set_ylabel('open')
    lns = line1 + line2 + line3 + line4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=2)
    plt.show()
    plt.savefig(path+"factor_sig.png")
    #plt.close()
    data.reset_index(inplace=True)

def get_newFreq_datetime(data_newfreq):
    try: 
        data_newfreq['date_time'] = data_newfreq.apply(lambda x:datetime.datetime.strptime\
            (str(int(x['date']))+' '+str(int(x['time']))[:-5],'%Y%m%d %H%M'), axis=1) 
    except KeyError: # 若报错，一般为缺乏 'time' 字段
        data_newfreq['date_time'] = data_newfreq.apply(lambda x:datetime.datetime.strptime\
            (str(int(x['date']))+' '+str(1500),'%Y%m%d %H%M'), axis=1) 
    
    return data_newfreq

# 给价格后复权，并保留未复权价格：后复权价 = 价格*后复权因子
def get_fuquan_data(data):
    '''
    Parameters
        data [dataframe]    数据(字段['factor'（复权因子）,'high','low','open','close'])
    '''
    col_list = ['high','low','open','close']
    for i in col_list:
        data['fq_'+ i] = np.multiply(data[i], data['factor'])
    data.drop(['factor'], axis=1, inplace=True) # 后面的因子也取名叫factor
    data['date_time'] = data.apply(lambda x:datetime.datetime.strptime\
            (str(int(x['date']))+' '+str(1500),'%Y%m%d %H%M'), axis=1) 
    return data

# 获取日度/年化收益率
def get_open_ret(value_data, pricename='open'):
    '''
    Parameters
        value     [dataframe] 价值/格数据[['open'(,'date_time'【year需要】)]
        pricename [str]       价格data中的价格列名 eg：'open'
    Return
        ret      [series]     年收益率 & 频率收益率
                 [dataframe]  日收益率，字段[['date_time','date','ret']]
    '''
    # 策略收益率
    #value_data.rename(columns={pricename:'price'},inplace=True)
    value_data = value_data.copy()

    value_data.loc[:,'year'] = value_data.loc[:,'date_time'].apply(lambda x:x.year)
    value_data.loc[:,'day'] = value_data.loc[:,'date_time'].apply(lambda x:x.year*10000+x.month*100+x.day)
    # groupby来调整数据频率
    first_price = value_data.loc[0,pricename]
    # 去年年底的价格（第一年则取第一天的值
    get_lastday = value_data.groupby(['year'])[pricename].nth(-1)
    get_lastday.name = 'lastyear_price'
    get_lastday = get_lastday.shift(1).fillna(first_price)
    value_data = pd.merge(value_data,get_lastday,left_on='year',right_index=True)

    data = value_data.copy()
    data = data.drop_duplicates(['day'])
    data['day_rank'] = data.groupby(['year'])['day'].rank()#value_data.groupby([['year','date_time']])
    value_data = pd.merge(value_data,data[['day','day_rank']],on='day',how='left')        
    #print('!!'*60)
    #print(value_data)
    # 求年化收益率:  [Pt/P(t-1) - 1] / (Tt-T(t-1)/244)
    # 开盘价--明天的开盘价比今天开盘价
    value_data['ret'] = np.divide( np.divide(value_data[pricename].shift(-1), value_data['lastyear_price'])-1,
                    ((value_data['day_rank'])/244))
    return value_data['ret']



if __name__ == '__main__':    
    index_future_code = {'IC':'000905.SH','IF':'000300.SH','IH':'000016.SH'}
    future_code = 'IC'; index_code = index_future_code[future_code]

    # 定义策略中需要用到的参数
    N = 60 # 滚动窗口
    s = 0.5 # 取值0.3-0.7； ±s 倍标准差作为开平仓阈值

    # 导入 数据（含复权因子 和 日度数据）
    filename = 'basic_data//'+str(future_code)+'_settle_factor_dlv_info.csv'
    data = pd.read_csv(filename, header = 0, index_col = 0)
    # 复权，并增加 一列 'datetime'
    data = get_fuquan_data(data)
    # print(data)

    # 指数成分股数据 & 成交额数据，给其降序排列
    index_data = pd.read_csv('basic_data//index_component//'+ index_code +'.csv', header = 0)
    amount_data = pd.read_csv('basic_data//amount.csv', header = 0)

    amount_data = get_amount_rank(index_data,amount_data)
    # 获取信号
    data_factor = get_factor(data,amount_data,N)
    
    ### 获取买卖信号
    data_factor = data_factor.reset_index()
    print(data_factor.describe())#['factor'].value_counts())
    
    data_sig = get_trading_sig(data_factor,s)
    
