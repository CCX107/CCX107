import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class UnifiedUserBehaviorCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols):#创建初始参数，后续都要使用，所以在__init__中使用
        self.feature_cols = feature_cols#定义我们最终要输出的特征列顺序，确保每次 transform 都输出一致的列
        self.night_weights = {h: np.exp(-((h - 3.0)**2) / 6.0) for h in range(24)}#构造晚上用户活跃权重，峰值在凌晨3点，向两边平滑衰减
        self.evening_weights = {h: np.exp(-((h - 21.0)**2) / 4.0) for h in range(24)}#构造傍晚用户活跃权重，峰值在晚上9点，向两边平滑衰减

    def fit(self, X, y=None):
        df = X.copy()#复制一份新的数据
        
        
        # 学习真人基准分布
        daily_hour = df.groupby(["date", "hour_time"]).size().reset_index(name="pv")#按照日期、小时分组，size()统计会比 count()快
        daily_hour["ratio"] = daily_hour["pv"] / (daily_hour.groupby("date")["pv"].transform("sum") + 1e-12)#计算每个小时的访问占当天总访问的比例，避免除零错误
        standard_dist = daily_hour.groupby("hour_time")["ratio"].median().clip(lower=1e-6)#取每个小时的中位数作为标准分布，clip 确保没有零值，避免后续计算中的 log(0) 问题
        self.ref_hour_dist_ = (standard_dist / standard_dist.sum()).to_dict()#归一化成概率分布，并转换成字典形式，方便后续映射
        
        self.ref_night_mass_ = sum(self.ref_hour_dist_.get(h, 0.0) * self.night_weights[h] for h in range(24))#计算基准的夜晚活跃质量总和，作为后续风险计算的分母基准
        self.ref_evening_mass_ = sum(self.ref_hour_dist_.get(h, 0.0) * self.evening_weights[h] for h in range(24))#计算基准的傍晚活跃质量总和，作为后续风险计算的分母基准

        # 学习标准 C 段聚集度
        df['ip_c_segment'] = df['$ip'].astype(str).str.rsplit('.', n=1).str[0]#获取c段ip
        ip_counts = df.groupby(['date', 'ip_c_segment'])['distinct_id'].nunique().reset_index(name='cnt')#统计每天每个c段的独立用户数，得到一个包含 date、ip_c_segment 和 cnt（独立用户数）的 DataFrame
        self.ref_c_mean_ = ip_counts.groupby('date')['cnt'].median().mean()#计算c段独立用户数的中位数的均值
        self.ref_c_std_ = ip_counts.groupby('date')['cnt'].std().mean() or 1.0#计算c段独立用户数的标准差的均值，或默认值1.0
        
        return self

    def transform(self, X):
        df = X.copy()#复制一份新的数据，避免修改原始输入
        gcols = ['date', 'distinct_id']#定义用户级别的分组列，后续很多特征都是基于用户维度来计算的
        
        
        
        # --- 0. 基础特征提取 ---
        df['ip_c_segment'] = df['$ip'].astype(str).str.rsplit('.', n=1).str[0]#获取c段ip
        df['is_direct'] = (df['$referrer'].isna() | (df['$referrer'] == "")).astype(int)#判断是否是直接访问
        
        user_pv_map = df.groupby(gcols, observed=True).size().reset_index(name='user_total_pv')#统计每个用户每天的总访问次数，得到一个包含 date、distinct_id 和 user_total_pv 的 DataFrame
        df = df.merge(user_pv_map, on=gcols, how='left')#把用户总访问次数合并回原始数据，方便后续特征计算
        df['is_1pv_user'] = (df['user_total_pv'] == 1).astype(int)#统计该用户是否是当天的单次访问用户，单次访问往往风险更高

        # ==========================================
        # 🏆 塔一：流量块风险 (Traffic Block Tower)
        # ==========================================
        block_stats = df.groupby(['date', '$url', 'hour_time'], observed=True).agg(
            block_pv=('distinct_id', 'count'),#统计每个网址每小时的访问次数，得到 block_pv
            block_uv=('distinct_id', 'nunique'),#统计每个网址每小时的独立用户数，得到 block_uv
            block_1pv_uv=('is_1pv_user', 'sum'),#统计每个网址每小时的单次访问用户数，得到 block_1pv_uv
            block_direct_pv=('is_direct', 'sum')#统计每个网址每小时的直接访问次数，得到 block_direct_pv
        ).reset_index()

        block_stats = block_stats.sort_values(['date', '$url', 'hour_time'])#按照日期、网址和小时排序，确保后续计算增长率时的顺序正确
        block_stats['prev_pv'] = block_stats.groupby(['date', '$url'])['block_pv'].shift(1).fillna(0)#统计前一个小时的访问次数，作为计算增长率的分母，fillna(0)处理第一条记录的缺失值
        block_stats['growth_rate'] = (block_stats['block_pv'] / (block_stats['prev_pv'] + 1e-6)).clip(upper=50)#计算增长率，clip 限制上限为50
        block_stats['uv_pv_ratio'] = block_stats['block_uv'] / (block_stats['block_pv'] + 1e-6)#计算独立用户数与访问次数的比率，这个比率过高可能意味着异常流量,范围在0-1之间，不用归一化
        block_stats['target_1pv_ratio'] = block_stats['block_1pv_uv'] / (block_stats['block_uv'] + 1e-6)#计算单次访问用户数与独立用户数的比率，这个比率过高可能意味着大量新用户涌入,范围在0-1之间，不用归一化
        block_stats['target_direct_ratio'] = block_stats['block_direct_pv'] / (block_stats['block_pv'] + 1e-6)#计算直接访问次数与总访问次数的比率，这个比率过高可能意味着大量用户是通过直接访问进入,范围在0-1之间，不用归一化

        df = df.merge(
            block_stats[['date', '$url', 'hour_time', 'block_pv', 'growth_rate', 'uv_pv_ratio', 'target_1pv_ratio', 'target_direct_ratio']],
            on=['date', '$url', 'hour_time'], how='left'
        )#合并流量块特征回原始数据，方便后续用户级别的风险计算

        df['is_suspicious_block'] = (
            (df['growth_rate'] > 3.0) & 
            (df['uv_pv_ratio'] > 0.9) & 
            (df['target_1pv_ratio'] > 0.2)
        ).astype(int)#定义一个强规则：如果某个网址某小时的访问增长率超过3倍，且独立用户占比超过90%，且单次访问用户占比超过20%，就认为这个流量块是可疑的，标记为1，否则为0

        # ==========================================
        # 🏰 塔二：个体时间与环境风险 (User Tower)
        # ==========================================
        exact_ip_uv = df.groupby(['date', '$ip'], observed=True)['distinct_id'].nunique().reset_index(name='exact_ip_uv')#统计每个 IP 每天的独立用户数，得到一个包含 date、$ip 和 exact_ip_uv 的 DataFrame，exact_ip_uv 过高可能意味着多个用户共享同一个 IP，存在风险
        c_seg_counts = df.groupby(['date', 'ip_c_segment'], observed=True)['distinct_id'].nunique().reset_index(name='cnt')#统计每个 C 段每天的独立用户数，得到一个包含 date、ip_c_segment 和 cnt 的 DataFrame，cnt 过高可能意味着大量用户来自同一个 C 段，存在风险
        c_seg_counts['c_zscore'] = (c_seg_counts['cnt'] - self.ref_c_mean_) / self.ref_c_std_#计算每个 C 段独立用户数的 z-score，衡量其相对于基准的异常程度，z-score 越高可能意味着风险越大
        
        df = df.merge(exact_ip_uv, on=['date', '$ip'], how='left')#合并 exact_ip_uv 特征回原始数据，方便后续用户级别的风险计算
        df = df.merge(c_seg_counts[['date', 'ip_c_segment', 'c_zscore']], on=['date', 'ip_c_segment'], how='left')#合并 c_zscore 特征回原始数据，方便后续用户级别的风险计算

        uh = df.groupby(gcols + ['hour_time'], observed=True).size().reset_index(name='pv')#统计每个用户每小时的访问次数，得到一个包含 date、distinct_id、hour_time 和 pv 的 DataFrame
        uh = uh.merge(user_pv_map, on=gcols, how='left')#合并用户总访问次数回用户小时级别的 DataFrame，方便计算用户在每个小时的访问占比
        uh['user_share'] = uh['pv'] / (uh['user_total_pv'] + 1e-12)#计算用户在每个小时的访问占比，避免除零错误
        uh['std_ratio'] = uh['hour_time'].map(self.ref_hour_dist_).fillna(1e-6)#填充标准比率，避免除零错误
        
        uh['kl_part'] = uh['user_share'] * np.log((uh['user_share'] + 1e-8) / (uh['std_ratio'] + 1e-8))#这个人的作息，到底在多大程度上偏离了正常人类
        uh['w_night_share'] = uh['user_share'] * uh['hour_time'].map(self.night_weights)#这个人的作息中，有多少是落在我们定义的凌晨高危时段的，权重越高代表越接近凌晨高危时段
        uh['w_evening_share'] = uh['user_share'] * uh['hour_time'].map(self.evening_weights)#这个人的作息中，有多少是落在我们定义的傍晚高危时段的，权重越高代表越接近傍晚高危时段

        hour_feat = uh.groupby(gcols, observed=True).agg(
            time_kl_dist=('kl_part', 'sum'),#计算用户作息分布与正常人类基准分布的 KL 散度，越大代表作息越异常
            user_night_density=('w_night_share', 'sum'),#计算用户在凌晨高危时段的加权访问占比，越大代表用户作息越集中在凌晨高危时段
            user_evening_density=('w_evening_share', 'sum')#计算用户在傍晚高危时段的加权访问占比，越大代表用户作息越集中在傍晚高危时段
        )
        hour_feat['night_relative_risk'] = (np.log(hour_feat['user_night_density'] + 1e-7) - np.log(self.ref_night_mass_ + 1e-7)).clip(lower=0)#看用户比大盘平均水平高出了多少个数量级
        hour_feat['evening_relative_risk'] = (np.log(hour_feat['user_evening_density'] + 1e-7) - np.log(self.ref_evening_mass_ + 1e-7)).clip(lower=0)#看用户比大盘平均水平高出了多少个数量级

        # ==========================================
        # 👯‍♂️ 塔三：克隆人攻击风险 (Clone Attack Tower) 
        # ==========================================
        user_signature = df.groupby(gcols, observed=True).agg(
            hour=('hour_time', 'mean'),#用户的平均访问小时，虽然不一定有实际意义，但可以作为用户作息的一个粗略特征
            url_nunique=('$url', 'nunique'),#用户访问过的不同网址数量，过少可能意味着用户行为单一，存在风险
            is_direct=('is_direct', 'mean'),#用户访问中直接访问的比例，过高可能意味着用户行为异常，存在风险
            ip_c=('ip_c_segment', 'first')#用户的 C 段 IP，虽然不一定有实际意义，但可以作为用户环境的一个粗略特征
        ).reset_index()

        # 统计相同类型用户
        user_signature['hour_bin'] = user_signature['hour'].round()#把用户的平均访问小时四舍五入到整数，作为一个粗略的时间特征，方便后续分组统计
        user_signature['cluster_size'] = user_signature.groupby(
            ['date', 'hour_bin', 'url_nunique', 'is_direct'], observed=True
        )['distinct_id'].transform('count')#计算每个用户群体的大小，即具有相同特征的用户数量

        # 转为按 date 和 distinct_id 为索引的形式，方便下一步无缝 join
        cluster_feat = user_signature.set_index(gcols)[['cluster_size']]
        # ==========================================
        # ⚔️ 终极融合与群体收网
        # ==========================================
        features = df.groupby(gcols, observed=True).agg(
            total_pv=('user_total_pv', 'max'),#统计每个用户最大 pv
            is_direct_ratio=('is_direct', 'mean'),#统计每个用户的直接访问比例，越高可能意味着用户行为越异常，存在风险
            max_c_zscore=('c_zscore', 'max'),#统计每个用户所在 C 段独立用户数 z-score 的最大值，越高可能意味着用户所在的 C 段越异常，存在风险
            max_exact_ip_uv=('exact_ip_uv', 'max'),#统计每个用户ip段独立用户是的最大值，越高可能意味着用户所在的 IP 越异常，存在风险
            max_block_pv=('block_pv', 'max'),#统计每个用户访问过的 URL 在对应小时的最大访问次数，越高可能意味着用户访问了一个非常热门的 URL，存在风险   
            avg_block_pv=('block_pv', 'mean'),#统计每个用户访问过的 URL 在对应小时的平均访问次数，越高可能意味着用户访问了一个非常热门的 URL，存在风险          
            max_growth_rate=('growth_rate', 'max'), #统计每个用户访问过的 URL 在对应小时的最大增长率，越高可能意味着用户访问了一个正在被攻击的 URL，存在风险 
            max_uv_pv_ratio=('uv_pv_ratio', 'max'), #统计每个用户访问过的 URL 在对应小时的最大独立用户数与访问次数的比率，越高可能意味着用户访问了一个独立用户占比异常的 URL，存在风险      
            max_target_1pv_ratio=('target_1pv_ratio', 'max'), #统计每个用户访问过的 URL 在对应小时的只访问一次的独立用户数与独立用户数的比率，越高可能意味着用户访问了一个单次访问用户占比异常的 URL，存在风险
            max_target_direct_ratio=('target_direct_ratio', 'max'), #统计每个用户访问过的 URL 在对应小时的直接访问次数与总访问次数的比率，越高可能意味着用户访问了一个直接访问占比异常的 URL，存在风险
            attack_block_ratio=('is_suspicious_block', 'mean')#统计用户访问的URL中可疑流量块的比例，越高可能意味着用户更倾向于访问可疑流量块，存在风险
            
        ).join(hour_feat).join(cluster_feat) #把之前计算的时间特征和克隆攻击特征合并到用户级别的特征表中，方便后续风险计算
        
        features['max_c_zscore'] = features['max_c_zscore'].clip(lower=0)#保持底层特征的干净

        # 基础双塔风险
        features['time_risk'] = features['time_kl_dist'] * np.log1p(features['total_pv'])#用户作息异常程度乘以访问次数的对数，访问次数越多，风险越大
        features['env_risk'] = np.log1p(features['max_exact_ip_uv']) * 0.5 + np.log1p(features['max_c_zscore']) * 0.5#用户环境异常程度乘以访问次数的对数，访问次数越多，风险越大
        features['final_time_risk'] = features[['time_risk', 'night_relative_risk', 'evening_relative_risk', 'env_risk']].max(axis=1)
      
        # 🎯 强规则 1: 流量块攻击狙击
        is_block_attack = (
            (features['attack_block_ratio'] > 0.7) & 
            (features['total_pv'] <= 3) & 
            (features['is_direct_ratio'] > 0.7)
        )
        features.loc[is_block_attack, 'final_time_risk'] += 25.0 
        
        # 🎯 强规则 2: 克隆人攻击狙击 (截图逻辑)
        is_clone_attack = (
            (features['cluster_size'] > 50) & 
            (features['total_pv'] <= 2)
        )
        features.loc[is_clone_attack, 'final_time_risk'] += 20.0

        # 补充凌晨高危区的基础拦截
        is_night_attack = (features['total_pv'] == 1) & (features['is_direct_ratio'] > 0.6) & (features['user_night_density'] > 0.4)
        features.loc[is_night_attack, 'final_time_risk'] += 15.0
        #补充晚上高危区的基础拦截
        is_evening_stealth_attack = (
        (features['user_evening_density'] > 0.5) &  # 傍晚密度极高
        (features['total_pv'] == 1) &               # 依然是单点攻击
        (features['is_direct_ratio'] == 1.0) &      # 必须是100%纯净的空降
        (features['cluster_size'] > 20)             # 关键：必须伴随一定的克隆特征
        )
        features.loc[is_evening_stealth_attack, 'final_time_risk'] += 10.0

        output_list = list(self.feature_cols) 
        
        # 第二步：强行把“业务分”加进输出名单。
        # 这样 reindex 就不会删掉它，而是把它一起吐给 Streamlit
        if 'final_time_risk' not in output_list:
            output_list.append('final_time_risk')
            
        # 第三步：现在 output_list 是 13 个名字了，安检通过！
        return features.reindex(columns=output_list, fill_value=0.0)