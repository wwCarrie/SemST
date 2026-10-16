import configparser

class Config(object):
    """配置加载类，从.ini里面读取超参数"""
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))

        # Parameter
        self.fdim = conf.getint("Data_Setting", "fdim") # 特征维度
        self.k = conf.getint("Model_Setup", "k") # 邻居数量
        self.radius = conf.getint("Model_Setup", "radius") # 半径
        self.seed = conf.getint("Model_Setup", "seed") # 随机种子
        self.lr = conf.getfloat("Model_Setup", "lr") # 学习率
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay") # 权重     衰减
        self.nhid1 = conf.getint("Model_Setup", "nhid1") # 第一层隐藏单元数
        self.nhid2 = conf.getint("Model_Setup", "nhid2") # 第二层隐藏单元数
        self.llm_modulation_ratio = conf.getfloat("Model_Setup", "llm_modulation_ratio") # LLM调制比率

        self.dropout = conf.getfloat("Model_Setup", "dropout") # Dropout比率
        self.epochs = conf.getint("Model_Setup", "epochs") # 训练轮数
        self.alpha = conf.getfloat("Model_Setup", "alpha") # Alpha参数
        self.beta = conf.getfloat("Model_Setup", "beta") # Beta参数
        self.gamma = conf.getfloat("Model_Setup", "gamma") # Gamma参数
        self.no_cuda = conf.getboolean("Model_Setup", "no_cuda") # 是否禁用CUDA
        self.no_seed = conf.getboolean("Model_Setup", "no_seed") # 是否禁用随机种子
