


class PortfolioPosition():

    _SIDE_LONG = "LONG"
    _SIDE_SHORT = "SHORT"
    _DEF_PORTF_AMT = 100000

    def __init__(self,p_symbol):
        self.symbol =p_symbol
        self.side=None
        self.date_open=None
        self.price_open=None
        self.date_close=None
        self.price_close=None


    def open_pos(self,side,date,price):
        self.side=side
        self.date_open=date
        self.price_open=float(price)


    def close_pos(self,date,price):

        self.date_close=date
        self.price_close=float(price)


    def calculate_pct_profit(self):
        if self.price_close is not None and self.price_open is not None:

            if self.side==PortfolioPosition._SIDE_LONG:
                if self.price_open >0:
                    return round( ((self.price_close-self.price_open)/self.price_open)*100,2)
                else:
                    raise Exception("Could not divide by 0 on price_open=0 for symbol {}".format(self.symbol))
            elif self.side == PortfolioPosition._SIDE_SHORT:
                    if self.price_open > 0:
                        return round(((self.price_open- self.price_close ) / self.price_open) * 100, 2)
                    else:
                        raise Exception("Could not divide by 0 on price_open=0 for symbol {}".format(self.symbol))
        else :
            raise Exception("Position for symbol {} not properly closed!  cannot calculate the pct profit".format(self.symbol))


    def calculate_th_nom_profit(self,portf_amt=_DEF_PORTF_AMT):
        pct_profit=self.calculate_pct_profit()
        return  (pct_profit/100)*portf_amt


