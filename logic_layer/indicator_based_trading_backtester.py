from business_entities.porft_summary import PortfSummary
from business_entities.portf_position import PortfolioPosition
from business_entities.portf_position_summary import PortfPositionSummary


class IndicatorBasedTradingBacktester:

    _INDIC_CLASSIF_COL="classification"
    _INDIC_CLASSIF_START_DATE_COL = "date_start"
    _INDIC_CLASSIF_END_DATE_COL = "date_end"

    def __init__(self):
        pass


    def __get_side__(self,indic_cassif_row,inv ):

        if inv  :
            if indic_cassif_row[IndicatorBasedTradingBacktester._INDIC_CLASSIF_COL]==PortfolioPosition._SIDE_LONG:
                return PortfolioPosition._SIDE_SHORT
            else:
                return PortfolioPosition._SIDE_LONG
        else:
            return indic_cassif_row[IndicatorBasedTradingBacktester._INDIC_CLASSIF_COL] #Nothing to switch


    def __switch_side__(self,side ):

        if side == PortfolioPosition._SIDE_LONG:
            return PortfolioPosition._SIDE_SHORT
        elif side == PortfolioPosition._SIDE_SHORT:
            return PortfolioPosition._SIDE_LONG
        else:
            raise Exception("Invalid side switching sides! : {}".format(side))

    def __get_date_price__(self,series_df,symbol,date):


        try:
            if date in series_df['date'].values:
                series_row = series_df.loc[series_df['date'] == date].iloc[0]
            else:
                series_row = series_df.sort_values(by='date', ascending=False).iloc[0]

            return series_row[symbol]

        except Exception as e:
            raise Exception("Could not find a price for symbol {} and date {}  on series_df!!!".format(symbol,date))

    def __calculate_portfolio_performance__(self,symbol,portf_positions_arr):

        summary=PortfSummary(symbol,PortfolioPosition._DEF_PORTF_AMT)

        for pos in portf_positions_arr:
            pct_profit=pos.calculate_pct_profit()
            th_nom_profit=pos.calculate_th_nom_profit()

            portf_pos_summary=PortfPositionSummary(pos,pct_profit,th_nom_profit,PortfolioPosition._DEF_PORTF_AMT)
            summary.append_position_summary(portf_pos_summary)

        return summary

    #NOTE:   Indicator based strategies consdier that the same price is used to close and open the next pos
    #This means. If closing a LONG position at 300.21, the system assumes that closing at 300.21 and goes SHORT at the same price
    def backtest_indicator_based_strategy(self,symbol,symbol_df,indic_classif_df,inv):
        last_pos=None
        portf_positions_arr=[]
        for index, row in indic_classif_df.iterrows():

            side = self.__get_side__(row, inv)
            if last_pos is None:
                last_pos = PortfolioPosition(symbol)
                open_date=row[IndicatorBasedTradingBacktester._INDIC_CLASSIF_START_DATE_COL]
                open_price = self.__get_date_price__(symbol_df, symbol, open_date)
                last_pos.open_pos(side, open_date, open_price)

            close_date=row[IndicatorBasedTradingBacktester._INDIC_CLASSIF_END_DATE_COL]
            close_price= self.__get_date_price__(symbol_df, symbol, close_date)
            last_pos.close_pos(close_date,close_price)
            portf_positions_arr.append(last_pos)

            side = self.__switch_side__(side)
            last_pos = PortfolioPosition(symbol)
            last_pos.open_pos(side, close_date, close_price)


        return self.__calculate_portfolio_performance__(symbol,portf_positions_arr)



