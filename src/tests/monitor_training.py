class TradeLogger:
    def __init__(self):
        self.trades = []
        
    def log_trade(self, trade_data):
        self.trades.append(trade_data)
        
    def log_episode_summary(self, summary_data):
        pass
