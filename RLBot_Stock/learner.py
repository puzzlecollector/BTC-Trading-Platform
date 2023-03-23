'''
import environment, agent, networks as well as other utility files
'''
class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()
    def __init__(self, rl_method="rl", stock_code=None, chart_data=None,
                 training_data = None, min_trading_price=100000, max_trading_price=10000000,
                 net="dnn", num_steps=1, lr=0.001,
                 discount_factor=0.9, num_epochs=1000,
                 balance=100000000, start_epsilon=1,
                 value_network=None, policty_network=None,
                 output_path="", reuse_models=True):
        assert min_trading_price > 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price
        assert num_steps > 0
        assert lr > 0
        self.rl_method = rl_method
        self.discount_factor = discount_factor
        self.num_epochs = num_epochs
        self.start_epsilon = start_epsilon
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environemnt = Environment(chart_data)
        self.agent = Agent(self.environment, balance, min_trading_price, max_trading_price)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        # self.visualizer = visualizer
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        # 에포크 관련 정보
        self.loss = 0
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        # 로그 등 출력 경로
        self.output_path = output_path
