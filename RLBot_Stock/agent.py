import numpy as np
import utils

class Agent:
    # 에이전트 상태가 구성하는 값 개수
    # 주식 보유 비율, 손익률, 주당 매수 단가 대부 주가 등락률
    STATE_DIM = 3

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015 # 거래 수수료 0.015%
    TRADING_TAX = 0.0025 # 거래세 0.25%

    # 행동
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS) # 인공 신경망에서 고려할 출력의 개수

    def __init__(self, environment, initial_balance, min_trading_price, max_trading_price):
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.enviroment = environment
        self.initial_balance = initial_balance # 초기 자본금

        # 최소 단일 매매 금액, 최대 단일 매매 금액
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        # Agent 클래스의 속성
        self.balance = initial_balance # 현재 현금 잔고
        self.num_stocks = 0 # 보유 주식 수
        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.num_buy = 0 # 매수 횟수
        self.num_sell = 0 # 매도 횟수
        self.num_hold = 0 # 관망 횟수

        # Agent 클래스의 상태
        self.ratio_hold = 0 # 주식 보유 비율
        self.profitloss = 0 # 손익률
        self.avg_buy_price = 0 # 주당 매수 단가

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks * self.environment.get_price() / self.portfolio_value
        return (self.ratio_hold,
                self.profitloss,
                (self.environment.get_price() / self.avg_buy_price) - 1 if self.avg_buy_price > 0 else 0)

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0
        pred = pred_policy
        if pred is None:
            pred = pred_value
        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            # value가 모두 같아서 어떤 결정을 내릴지 모르겠다
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = 0.5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE):
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_price
        added_trading_price = max(min(
            int(confidence * (self.max_trading_price - self.min_trading_price)),
            self.max_trading_price - self.min_trading_price), 0)
        ))
        trading_price = self.min_tading_price + added_trading_price
        return max(int(trading_price / self.environment.get_price()), 1)

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()
