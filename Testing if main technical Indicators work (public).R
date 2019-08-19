#backtesting certain technical indicators to see iXLB they work

library(quantmod)
library(PerformanceAnalytics)
library(tidyverse)

#download symbols testing
setDefaults(getSymbols.av, api.key = "yourkeyhere")

Stock_symbols <- c("SPY",
             "AAPL",
             "GE",
             "XLB",
             "F")

getSymbols(Stock_symbols, src = "av", 
           output.size = "full", adjusted = TRUE)

SPY$MACD <- MACD(SPY$SPY.Adjusted, nFast = 12, nSlow = 26, nSig = 9, maType = "EMA")
autoplot(SPY[,c(7,8)]["2018"], facets = NULL)
SPY$Signal <- ifelse(SPY$macd>SPY$MACD,1,-1)
tail(SPY)
SPY$Returns <- Return.calculate(SPY$SPY.Adjusted)
SPY$Returns_lagged <- lag.xts(SPY$Returns, -1)
SPY$Portfolio_Returns <- SPY$Returns_lagged*SPY$Signal
chart.CumReturns(SPY[,c(10,12)])
table.AnnualizedReturns(SPY[,c(10,12)])

#Curious if a long/flat would work instead
SPY$Signal_long_flat <- ifelse(SPY$macd>SPY$MACD,1,0)
SPY$Portfolio_Returns_long_flat <- SPY$Returns_lagged*SPY$Signal_long_flat
chart.CumReturns(SPY[,c(10,14)])
table.AnnualizedReturns(SPY[,c(10,14)])

#long term signal - more in line with efficiacy of long term trend following
SPY$MACD_LT <- MACD(SPY$SPY.Adjusted, nFast = 12*10, nSlow = 26*10, nSig = 9*10, maType = "EMA")
SPY$Signal_long_flat_LT <- ifelse(SPY$macd.1>SPY$MACD_LT,1,0)
SPY$Portfolio_Returns_long_flat_LT <- SPY$Returns_lagged*SPY$Signal_long_flat_LT
chart.CumReturns(SPY[,c(10,18)])
table.AnnualizedReturns(SPY[,c(10,18)])
ncol(SPY)

#well long flat did better, but still buy and hold got the upper hand, and the longer the signal
#the better as a trend following indicator

AAPL$MACD <- MACD(AAPL$AAPL.Adjusted, nFast = 12, nSlow = 26, nSig = 9, maType = "EMA")
autoplot(AAPL[,c(7,8)]["2018"], facets = NULL)
AAPL$Signal <- ifelse(AAPL$macd>AAPL$MACD,1,-1)
tail(AAPL)
AAPL$Returns <- Return.calculate(AAPL$AAPL.Adjusted)
AAPL$Returns_lagged <- lag.xts(AAPL$Returns, -1)
AAPL$Portfolio_Returns <- AAPL$Returns_lagged*AAPL$Signal
chart.CumReturns(AAPL[,c(10,12)])
table.AnnualizedReturns(AAPL[,c(10,12)])

#Curious is a long/flat would work instead
AAPL$Signal_long_flat <- ifelse(AAPL$macd>AAPL$MACD,1,0)
AAPL$Portfolio_Returns_long_flat <- AAPL$Returns_lagged*AAPL$Signal_long_flat
chart.CumReturns(AAPL[,c(10,14)])
table.AnnualizedReturns(AAPL[,c(10,14)])

#long term signal - more in line with efficacy of long term trend following
AAPL$MACD_LT <- MACD(AAPL$AAPL.Adjusted, nFast = 12*10, nSlow = 26*10, nSig = 9*10, maType = "EMA")
AAPL$Signal_long_flat_LT <- ifelse(AAPL$macd.1>AAPL$MACD_LT,1,0)
AAPL$Portfolio_Returns_long_flat_LT <- AAPL$Returns_lagged*AAPL$Signal_long_flat_LT
chart.CumReturns(AAPL[,c(10,18)])
table.AnnualizedReturns(AAPL[,c(10,18)])

#Interesting to see how the signal helped again better with long/flat 
#the shorter MACD did better than Long term with AAPL and opposite with SPY

GE$MACD <- MACD(GE$GE.Adjusted, nFast = 12, nSlow = 26, nSig = 9, maType = "EMA")
autoplot(GE[,c(7,8)]["2018"], facets = NULL)
GE$Signal <- ifelse(GE$macd>GE$MACD,1,-1)
GE$Returns <- Return.calculate(GE$GE.Adjusted)
GE$Returns_lagged <- lag.xts(GE$Returns, -1)
GE$Portfolio_Returns <- GE$Returns_lagged*GE$Signal
chart.CumReturns(GE[,c(10,12)])
table.AnnualizedReturns(GE[,c(10,12)])

#short term MACD certainly didn't work

#Curious is a long/flat would work instead
GE$Signal_long_flat <- ifelse(GE$macd>GE$MACD,1,0)
GE$Portfolio_Returns_long_flat <- GE$Returns_lagged*GE$Signal_long_flat
chart.CumReturns(GE[,c(10,14)])
table.AnnualizedReturns(GE[,c(10,14)])

#much better


#long term signal - more in line with efficacy of long term trend following
GE$MACD_LT <- MACD(GE$GE.Adjusted, nFast = 12*10, nSlow = 26*10, nSig = 9*10, maType = "EMA")
GE$Signal_long_flat_LT <- ifelse(GE$macd.1>GE$MACD_LT,1,0)
GE$Portfolio_Returns_long_flat_LT <- GE$Returns_lagged*GE$Signal_long_flat_LT
chart.CumReturns(GE[,c(10,18)])
table.AnnualizedReturns(GE[,c(10,18)])

#that worked a lot better, and we are seeing that if the price will exhibit
#a strong long term uptrend, you want to have a slow long term signal
#but if has rollercoaster like AAPL you want a shorter time frame signal

F$MACD <- MACD(F$F.Adjusted, nFast = 12, nSlow = 26, nSig = 9, maType = "EMA")
autoplot(F[,c(7,8)]["2018"], Facets = NULL)
F$Signal <- iFelse(F$macd>F$MACD,1,-1)
F$Returns <- Return.calculate(F$F.Adjusted)
F$Returns_lagged <- lag.xts(F$Returns, -1)
F$PortFolio_Returns <- F$Returns_lagged*F$Signal
chart.CumReturns(F[,c(10,12)])
table.AnnualizedReturns(F[,c(10,12)])

#short term MACD certainly didn't work

#Curious is a long/Flat would work instead
F$Signal_long_Flat <- iFelse(F$macd>F$MACD,1,0)
F$PortFolio_Returns_long_Flat <- F$Returns_lagged*F$Signal_long_Flat
chart.CumReturns(F[,c(10,14)])
table.AnnualizedReturns(F[,c(10,14)])

#long/Flat keeps outperForming

#long term signal - more in line with efficacy of long term trend Following
F$MACD_LT <- MACD(F$F.Adjusted, nFast = 12*10, nSlow = 26*10, nSig = 9*10, maType = "EMA")
F$Signal_long_Flat_LT <- iFelse(F$macd.1>F$MACD_LT,1,0)
F$PortFolio_Returns_long_Flat_LT <- F$Returns_lagged*F$Signal_long_Flat_LT
chart.CumReturns(F[,c(10,18)])
table.AnnualizedReturns(F[,c(10,18)])

#since Ford didn't have a steady uptrend we see the same thing with AAPL

XLB$MACD <- MACD(XLB$XLB.Adjusted, nXLBast = 12, nSlow = 26, nSig = 9, maType = "EMA")
autoplot(XLB[,c(7,8)]["2018"], facets = NULL)
XLB$Signal <- ifelse(XLB$macd>XLB$MACD,1,-1)
XLB$Returns <- Return.calculate(XLB$XLB.Adjusted)
XLB$Returns_lagged <- lag.xts(XLB$Returns, -1)
XLB$Portfolio_Returns <- XLB$Returns_lagged*XLB$Signal
chart.CumReturns(XLB[,c(10,12)])
table.AnnualizedReturns(XLB[,c(10,12)])

#short term MACD certainly didn't work

#Curious is a long/Flat would work instead
XLB$Signal_long_flat <- ifelse(XLB$macd>XLB$MACD,1,0)
XLB$Portfolio_Returns_long_flat <- XLB$Returns_lagged*XLB$Signal_long_flat
chart.CumReturns(XLB[,c(10,14)])
table.AnnualizedReturns(XLB[,c(10,14)])

#long/flat keeps outperforming

#long term signal - more in line with efficacy of long term trend following
XLB$MACD_LT <- MACD(XLB$XLB.Adjusted, nfast = 12*10, nSlow = 26*10, nSig = 9*10, maType = "EMA")
XLB$Signal_long_flat_LT <- ifelse(XLB$macd.1>XLB$MACD_LT,1,0)
XLB$Portfolio_Returns_long_flat_LT <- XLB$Returns_lagged*XLB$Signal_long_flat_LT
chart.CumReturns(XLB[,c(10,18)])
table.AnnualizedReturns(XLB[,c(10,18)])

#still see that with long term uptrend did better with long term long/flat

XLE$MACD <- MACD(XLE$XLE.Adjusted, nXLEast = 12, nSlow = 26, nSig = 9, maType = "EMA")
autoplot(XLE[,c(7,8)]["2018"], facets = NULL)
XLE$Signal <- ifelse(XLE$macd>XLE$MACD,1,-1)
XLE$Returns <- Return.calculate(XLE$XLE.Adjusted)
XLE$Returns_lagged <- lag.xts(XLE$Returns, -1)
XLE$Portfolio_Returns <- XLE$Returns_lagged*XLE$Signal
chart.CumReturns(XLE[,c(10,12)])
table.AnnualizedReturns(XLE[,c(10,12)])

#short term MACD certainly didn't work

#Curious is a long/flat would work instead
XLE$Signal_long_flat <- ifelse(XLE$macd>XLE$MACD,1,0)
XLE$Portfolio_Returns_long_flat <- XLE$Returns_lagged*XLE$Signal_long_flat
chart.CumReturns(XLE[,c(10,14)])
table.AnnualizedReturns(XLE[,c(10,14)])

#long/flat keeps outperforming

#long term signal - more in line with efficacy of long term trend following
XLE$MACD_LT <- MACD(XLE$XLE.Adjusted, nfast = 12*10, nSlow = 26*10, nSig = 9*10, maType = "EMA")
XLE$Signal_long_flat_LT <- ifelse(XLE$macd.1>XLE$MACD_LT,1,0)
XLE$Portfolio_Returns_long_flat_LT <- XLE$Returns_lagged*XLE$Signal_long_flat_LT
chart.CumReturns(XLE[,c(10,18)])
table.AnnualizedReturns(XLE[,c(10,18)])

#same old same old

XLF$MACD <- MACD(XLF$XLF.Adjusted, nXLFast = 12, nSlow = 26, nSig = 9, maType = "EMA")
autoplot(XLF[,c(7,8)]["2018"], facets = NULL)
XLF$Signal <- ifelse(XLF$macd>XLF$MACD,1,-1)
XLF$Returns <- Return.calculate(XLF$XLF.Adjusted)
XLF$Returns_lagged <- lag.xts(XLF$Returns, -1)
XLF$Portfolio_Returns <- XLF$Returns_lagged*XLF$Signal
chart.CumReturns(XLF[,c(10,12)])
table.AnnualizedReturns(XLF[,c(10,12)])

#short term MACD certainly didn't work

#Curious is a long/Flat would work instead
XLF$Signal_long_flat <- ifelse(XLF$macd>XLF$MACD,1,0)
XLF$Portfolio_Returns_long_flat <- XLF$Returns_lagged*XLF$Signal_long_flat
chart.CumReturns(XLF[,c(10,14)])
table.AnnualizedReturns(XLF[,c(10,14)])

#long/flat keeps outperforming

#long term signal - more in line with long term trend following
XLF$MACD_LT <- MACD(XLF$XLF.Adjusted, nfast = 12*10, nSlow = 26*10, nSig = 9*10, maType = "EMA")
XLF$Signal_long_flat_LT <- ifelse(XLF$macd.1>XLF$MACD_LT,1,0)
XLF$Portfolio_Returns_long_flat_LT <- XLF$Returns_lagged*XLF$Signal_long_flat_LT
chart.CumReturns(XLF[,c(10,18)])
table.AnnualizedReturns(XLF[,c(10,18)])

###

XLI$MACD <- MACD(XLI$XLI.Adjusted, nXLIast = 12, nSlow = 26, nSig = 9, maType = "EMA")
autoplot(XLI[,c(7,8)]["2018"], facets = NULL)
XLI$Signal <- ifelse(XLI$macd>XLI$MACD,1,-1)
XLI$Returns <- Return.calculate(XLI$XLI.Adjusted)
XLI$Returns_lagged <- lag.xts(XLI$Returns, -1)
XLI$Portfolio_Returns <- XLI$Returns_lagged*XLI$Signal
chart.CumReturns(XLI[,c(10,12)])
table.AnnualizedReturns(XLI[,c(10,12)])

#short term MACD certainly didn't work

#Curious is a long/flat would work instead
XLI$Signal_long_flat <- ifelse(XLI$macd>XLI$MACD,1,0)
XLI$Portfolio_Returns_long_flat <- XLI$Returns_lagged*XLI$Signal_long_flat
chart.CumReturns(XLI[,c(10,14)])
table.AnnualizedReturns(XLI[,c(10,14)])

#long/flat keeps outperforming. short term MACD long flat actually had better Sharpe Ratio

#long term signal - more in line with long term trend following
XLI$MACD_LT <- MACD(XLI$XLI.Adjusted, nfast = 12*10, nSlow = 26*10, nSig = 9*10, maType = "EMA")
XLI$Signal_long_flat_LT <- ifelse(XLI$macd.1>XLI$MACD_LT,1,0)
XLI$Portfolio_Returns_long_flat_LT <- XLI$Returns_lagged*XLI$Signal_long_flat_LT
chart.CumReturns(XLI[,c(10,18)])
table.AnnualizedReturns(XLI[,c(10,18)])

####

XLP$MACD <- MACD(XLP$XLP.Adjusted, nXLPast = 12, nSlow = 26, nSig = 9, maType = "EMA")
autoplot(XLP[,c(7,8)]["2018"], facets = NULL)
XLP$Signal <- ifelse(XLP$macd>XLP$MACD,1,-1)
XLP$Returns <- Return.calculate(XLP$XLP.Adjusted)
XLP$Returns_lagged <- lag.xts(XLP$Returns, -1)
XLP$Portfolio_Returns <- XLP$Returns_lagged*XLP$Signal
chart.CumReturns(XLP[,c(10,12)])
table.AnnualizedReturns(XLP[,c(10,12)])

#short term MACD certainly didn't work

#Curious is a long/flat would work instead
XLP$Signal_long_flat <- ifelse(XLP$macd>XLP$MACD,1,0)
XLP$Portfolio_Returns_long_flat <- XLP$Returns_lagged*XLP$Signal_long_flat
chart.CumReturns(XLP[,c(10,14)])
table.AnnualizedReturns(XLP[,c(10,14)])

#long/flat keeps outperforming. short term MACD long flat actually had better Sharpe Ratio

#long term signal - more in line with long term trend following
XLP$MACD_LT <- MACD(XLP$XLP.Adjusted, nfast = 12*10, nSlow = 26*10, nSig = 9*10, maType = "EMA")
XLP$Signal_long_flat_LT <- ifelse(XLP$macd.1>XLP$MACD_LT,1,0)
XLP$Portfolio_Returns_long_flat_LT <- XLP$Returns_lagged*XLP$Signal_long_flat_LT
chart.CumReturns(XLP[,c(10,18)])
table.AnnualizedReturns(XLP[,c(10,18)])
chart.Drawdown(XLP[,c(10,18)])

#same story, but interesting that the long term MACD long flat had better Sharpe

XLV$MACD <- MACD(XLV$XLV.Adjusted, nXLVast = 12, nSlow = 26, nSig = 9, maType = "EMA")
autoplot(XLV[,c(7,8)]["2018"], facets = NULL)
XLV$Signal <- ifelse(XLV$macd>XLV$MACD,1,-1)
XLV$Returns <- Return.calculate(XLV$XLV.Adjusted)
XLV$Returns_lagged <- lag.xts(XLV$Returns, -1)
XLV$Portfolio_Returns <- XLV$Returns_lagged*XLV$Signal
chart.CumReturns(XLV[,c(10,12)])
table.AnnualizedReturns(XLV[,c(10,12)])

#short term MACD certainly didn't work

#Curious is a long/flat would work instead
XLV$Signal_long_flat <- ifelse(XLV$macd>XLV$MACD,1,0)
XLV$Portfolio_Returns_long_flat <- XLV$Returns_lagged*XLV$Signal_long_flat
chart.CumReturns(XLV[,c(10,14)])
table.AnnualizedReturns(XLV[,c(10,14)])

#long/flat keeps outperforming. short term MACD long flat actually had better Sharpe Ratio

#long term signal - more in line with long term trend following
XLV$MACD_LT <- MACD(XLV$XLV.Adjusted, nfast = 12*10, nSlow = 26*10, nSig = 9*10, maType = "EMA")
XLV$Signal_long_flat_LT <- ifelse(XLV$macd.1>XLV$MACD_LT,1,0)
XLV$Portfolio_Returns_long_flat_LT <- XLV$Returns_lagged*XLV$Signal_long_flat_LT
chart.CumReturns(XLV[,c(10,18)])
table.AnnualizedReturns(XLV[,c(10,18)])
chart.Drawdown(XLV[,c(10,18)])

XLY$MACD <- MACD(XLY$XLY.Adjusted, nXLYast = 12, nSlow = 26, nSig = 9, maType = "EMA")
autoplot(XLY[,c(7,8)]["2018"], facets = NULL)
XLY$Signal <- ifelse(XLY$macd>XLY$MACD,1,-1)
XLY$Returns <- Return.calculate(XLY$XLY.Adjusted)
XLY$Returns_lagged <- lag.xts(XLY$Returns, -1)
XLY$Portfolio_Returns <- XLY$Returns_lagged*XLY$Signal
chart.CumReturns(XLY[,c(10,12)])
table.AnnualizedReturns(XLY[,c(10,12)])

#short term MACD certainly didn't work

#Curious is a long/flat would work instead
XLY$Signal_long_flat <- ifelse(XLY$macd>XLY$MACD,1,0)
XLY$Portfolio_Returns_long_flat <- XLY$Returns_lagged*XLY$Signal_long_flat
chart.CumReturns(XLY[,c(10,14)])
table.AnnualizedReturns(XLY[,c(10,14)])

#long/flat keeps outperforming. short term MACD long flat actually had better Sharpe Ratio

#long term signal - more in line with long term trend following
XLY$MACD_LT <- MACD(XLY$XLY.Adjusted, nfast = 12*10, nSlow = 26*10, nSig = 9*10, maType = "EMA")
XLY$Signal_long_flat_LT <- ifelse(XLY$macd.1>XLY$MACD_LT,1,0)
XLY$Portfolio_Returns_long_flat_LT <- XLY$Returns_lagged*XLY$Signal_long_flat_LT
chart.CumReturns(XLY[,c(10,18)])
table.AnnualizedReturns(XLY[,c(10,18)])

#With the use of up to 20 years of data for a few stocks and different industries
#we see that MACD short term long flat can really help if there is not going to be a steady
#up trend in a stock but the long term MACD long/flat generally did the best for MACD
#but the buy and hold did better for the general uptrend symbols

####################Testing SMA and ADX#############################################

#load or download data from Kenneth French website
load("~/Coding/R/Working Directory/Kenneth French Factor Data/10 US Industries Portfolios and Recession and Expansion Dates.RData")

head(FF_Industry_Portfolios_1)

#creating index of Industry Data
FF_Industry_Data_Index <- cumprod(1 + (FF_Industry_Portfolios_1$`Non Durables`)) - 1
FF_Industry_Data_Index$Durables <- cumprod(1 + (FF_Industry_Portfolios_1$Durables)) - 1
FF_Industry_Data_Index$Manufacturing <- cumprod(1 + (FF_Industry_Portfolios_1$Manufacturing)) - 1
FF_Industry_Data_Index$Energy <- cumprod(1 + (FF_Industry_Portfolios_1$Energy)) - 1
FF_Industry_Data_Index$HiTech <- cumprod(1 + (FF_Industry_Portfolios_1$`Hi-Tech`)) - 1
FF_Industry_Data_Index$Telecommunications <- cumprod(1 + (FF_Industry_Portfolios_1$Telecommunications)) - 1
FF_Industry_Data_Index$Shopping_Retail <- cumprod(1 + (FF_Industry_Portfolios_1$`Shopping/Retail`)) - 1
FF_Industry_Data_Index$HealthCare <- cumprod(1 + (FF_Industry_Portfolios_1$Healthcare)) - 1
FF_Industry_Data_Index$Utilities <- cumprod(1 + (FF_Industry_Portfolios_1$Utilities)) - 1
FF_Industry_Data_Index$Other <- cumprod(1 + (FF_Industry_Portfolios_1$Other)) - 1

FF_3_Market_Index <- cumprod(1 + (FF_3_Factor_and_Momentum$MKT_Plus_Rf)) - 1

#from Rblogger, the big nice thing is the performance function
#https://www.r-bloggers.com/parameter-optimization-for-strategy-2/

#2-Calculate the 200-Day SMAs:
smaHi200=SMA(FF_3_Market_Index["1990/"],200)
plot(smaHi200)

#3-Calculate the lagged trading signal vector:
binVec=lag(ifelse(FF_3_Market_Index["1990/"]>smaHi200,1,0))
tail(FF_3_Market_Index)
tail(smaHi200) 
tail(binVec)
#4-Get rid of the NAs:
binVec[is.na(binVec)]=0
                                                 
#5-Calculate returns vector and multiply out the trading vector with the returns vector to get the strategy return:
rets=Return.calculate(FF_3_Market_Index)
stratRets=binVec*rets
stratRets_1950_on <- stratRets["1950/"]

#6-Run performance analytics:
charts.PerformanceSummary(merge(rets,stratRets_1950_on["1990/"]))

Performance <- function(x) {
  
  cumRetx = Return.cumulative(x)
  annRetx = Return.annualized(x, scale=252)
  sharpex = SharpeRatio.annualized(x, scale=252)
  winpctx = length(x[x > 0])/length(x[x != 0])
  annSDx = sd.annualized(x, scale=252)
  
  DDs <- findDrawdowns(x)
  maxDDx = min(DDs$return)
  maxLx = max(DDs$length)
  
  Perf = c(cumRetx, annRetx, sharpex, winpctx, annSDx, maxDDx, maxLx)
  names(Perf) = c("Cumulative Return", "Annual Return","Annualized Sharpe Ratio",
                  "Win %", "Annualized Volatility", "Maximum Drawdown", "Max Length Drawdown")
  return(Perf)
}

Performance(stratRets_1950_on)

optimizeSMA(FF_Industry_Data_Index$Non.Durables["1990/"],FF_Industry_Portfolios_1$`Non Durables`["1990/"],smaInit = 3, smaEnd = 200)

#Optimizing the Parameter based on Constraint from rblogger
#https://www.r-bloggers.com/parameter-optimization-for-strategy-2/

#From the website

#My aim is to find out which SMA is the best to use for going long, and which SMA is the 
#best to use for going short on the S&P 500. Ideally, I should optimize the short SMA 
#for each long SMA (or vice-versa) to find the best combination, but I don’t think 
#optimizing them independently (as I did here) would make much of a difference 
#in this case. 

optimizeSMA=function(mainVector,returnsVector,smaInit=3,smaEnd=200,long=TRUE){
  
  bestSMA=0
  bestSharpe=0
  
  for( i in smaInit:smaEnd){
    smaVec=SMA(mainVector,i)
    if(long==T){
      
      binVec=lag(ifelse(mainVector>smaVec,1,0),1)
      binVec[is.na(binVec)]=0
      stratRets=binVec*returnsVector
      sharpe=SharpeRatio.annualized(stratRets, scale=252)
      if(sharpe>bestSharpe){
        bestSMA=i
        bestSharpe=sharpe
      }
      
    }else{
      
      binVec=lag(ifelse(mainVector,-1,0),1)
      binVec[is.na(binVec)]=0
      stratRets=binVec*returnsVector
      sharpe=SharpeRatio.annualized(stratRets, scale=252)
      if(sharpe>bestSharpe){
        bestSMA=i
        bestSharpe=sharpe
      }
    }
  }
  
  print(cbind(bestSMA, bestSharpe))
}

########Testing Money Flow Slope and Money Flow Index###########

getSymbols(c("SPY","GE"), src = "av", 
          output.size = "full", adjusted = TRUE)

SPY$MFI <- MFI(SPY[,c(2:4)], SPY$SPY.Volume, n = 14)
autoplot(SPY[,c(7:8)]["2018/"], facets = NULL)
SPY$MFISlope <- SMA(SPY$MFI, n = 14)

#create the signal for SMA 200 strategy
#this signal says if the price is above the trend, if not sell short
signal <- ifelse(SPY$MFISlope < SPY$MFI, 1, -1)

#add slippage and transaction costs in percentage term including slippage and commissions per Philisophical Economics on historical costs
slippage_costs <- -.0060

#add the signal and the trend strategy returns
SPY$signal <- signal

#add slippage and commision costs signal
SPY$slippage_signal <- lag(ifelse(SPY$signal != lag(SPY$signal, 1), slippage_costs, 0), k = -1)


#calculate the returns based on the price data times the signals. must add to signal a day to buy on the day
#after the signal
SPY$portfolio.return_no_costs <- (ROC(SPY$SPY.Adjusted)*lag(signal))
SPY$portfolio.return_withcosts <- SPY$portfolio.return_no_costs + SPY$slippage_signal

SPY$returns <- Return.calculate(SPY$SPY.Adjusted)
returns_MFSLOPE_LONG_FLAT <- table.AnnualizedReturns(SPY[,c(10,11,12)])
chart.CumReturns(SPY[,c(10,11,12)])

returns_MFSLOPE_LONG_SHORT <- table.AnnualizedReturns(SPY[,c(10,11,12)])
chart.CumReturns(SPY[,c(10,11,12)])

#with no costs the strategy only worked during 1999 -2005 but not after, and quite miserably
#and with costs it would be atrocious.

SPY2 <- SPY[,c(1:7)]

signal2 <- ifelse(SPY$MFI > 80, -1, ifelse(SPY$MFI < 20, 1, 0))
SPY2$signal <- signal2

SPY2$portfolio.return_no_costs <- (ROC(SPY2$SPY.Adjusted)*lag(signal2))
SPY2$returns <- Return.calculate(SPY2$SPY.Adjusted)
table.AnnualizedReturns(SPY2[,c(9:10)])
chart.CumReturns(SPY2[,c(9:10)])

#with the standard 80,20 threshold for the MFI it did have a higher sharpe and lower vol (substantially)
#vs passive but this is before costs

#after costs
slippage_costs <- -.0060
SPY2$slippage_signal <- lag(ifelse(SPY2$signal != lag(SPY2$signal, 1), slippage_costs, 0), k = -1)
SPY2$portfolio.return_withcosts <- SPY2$portfolio.return_no_costs + SPY2$slippage_signal

table.AnnualizedReturns(SPY2[,c(9,10,12)])
chart.CumReturns(SPY2[,c(9,10,12)])

#really bad after costs, losing strategy by itself

########Testing Money Flow Slope and Money Flow Index###########

getSymbols("GE", src = "av", 
           output.size = "full", adjusted = TRUE)

GE$MFI <- MFI(GE[,c(2:4)], GE$GE.Volume, n = 14)
GE$MFISlope <- SMA(GE$MFI, n = 14)

#create the signal 
#this signal says if the price is above the trend, if not sell short
signal <- ifelse(GE$MFISlope < GE$MFI, 1, -1)

#add slippage and transaction costs in percentage term including slippage and commissions per Philisophical Economics on historical costs
slippage_costs <- -.0060

#add the signal and the trend strategy returns
GE$signal <- signal

#add slippage and commision costs signal
GE$slippage_signal <- lag(ifelse(GE$signal != lag(GE$signal, 1), slippage_costs, 0), k = -1)


#calculate the returns based on the price data times the signals. must add to signal a day to buy on the day
#after the signal
GE$portfolio.return_no_costs <- (ROC(GE$GE.Adjusted)*lag(signal))
GE$portfolio.return_withcosts <- GE$portfolio.return_no_costs + GE$slippage_signal

GE$returns <- Return.calculate(GE$GE.Adjusted)
table.AnnualizedReturns(GE[,c(11,12,13)])
chart.CumReturns(GE[,c(11,12,13)], legend.loc = "topleft")

returns_MFSLOPE_LONG_SHORT <- table.AnnualizedReturns(GE[,c(10,11,12)])
chart.CumReturns(GE[,c(10,11,12)])

#with no costs the strategy did well but with costs at 60 bps slippage not good, 
#and ignores cost to borrow GE stock, which is currenlty .25% on IB

GE2 <- GE[,c(1:7)]

signal2 <- ifelse(GE$MFI > 80, -1, ifelse(GE$MFI < 20, 1, 0))
GE2$signal <- signal2

GE2$portfolio.return_no_costs <- (ROC(GE2$GE.Adjusted)*lag(signal2))
GE2$returns <- Return.calculate(GE2$GE.Adjusted)
table.AnnualizedReturns(GE2[,c(9:10)])
chart.CumReturns(GE2[,c(9:10)])

#MFI did add value on return and risk vs. passive before costs 

#after costs
slippage_costs <- -.0060
GE2$slippage_signal <- lag(ifelse(GE2$signal != lag(GE2$signal, 1), slippage_costs, 0), k = -1)
GE2$portfolio.return_withcosts <- GE2$portfolio.return_no_costs + GE2$slippage_signal

table.AnnualizedReturns(GE2[,c(9,10,12)])
chart.CumReturns(GE2[,c(9,10,12)])

#But not after costs, you would need to get costs down.

#I am seeing a trend with technical indicators in that if the security has wide oscillations and
#prolonged uptrend then using a single indicator can be useful but no always after costs. But if
#there is a solid long term uptrend like SPY, AAPL, AMZN, then stay away from anything but simple SMA, 
#which will even underperform.

#####On Balance Volume#######################################
#to confirm trend or imply reversal of trend#####
SPY
SPY$OBV <- OBV(SPY$SPY.Adjusted,SPY$SPY.Volume)
autoplot(SPY$OBV)

SPY$SMA <- SMA(SPY$SPY.Adjusted, 200)

#create the signal for SMA 200 strategy
#this signal says if the price is above the trend, if not sell short
signal <- ifelse(SPY$SPY.Adjusted > SPY$SMA & SPY$OBV > 0, 1,
                 ifelse(SPY$SPY.Adjusted < SPY$SMA & SPY$OBV < 0,-1, 0))

#add slippage and transaction costs in percentage term including slippage and commissions per Philisophical Economics on historical costs
slippage_costs <- -.0060

#add the signal and the trend strategy returns
SPY$signal <- signal

#add slippage and commision costs signal
SPY$slippage_signal <- lag(ifelse(SPY$signal != lag(SPY$signal, 1), slippage_costs, 0), k = -1)


#calculate the returns based on the price data times the signals. must add to signal a day to buy on the day
#after the signal
SPY$portfolio.return_no_costs <- (ROC(SPY$SPY.Adjusted)*lag(signal))
SPY$portfolio.return_withcosts <- SPY$portfolio.return_no_costs + SPY$slippage_signal
SPY$Passive_Returns <- Return.calculate(SPY$SPY.Adjusted)

table.AnnualizedReturns(SPY[,c(11:13)])
chart.CumReturns(SPY[,c(11:13)], legend.loc = "topleft")

#lets see if the OBV + SMA is better than simple SMA
signal2 <- ifelse(SPY$SPY.Adjusted < SPY$SMA, -1, 1)
SPY$signal2 <- signal2

#add slippage and commision costs signal
SPY$slippage_signal2 <- lag(ifelse(SPY$signal2 != lag(SPY$signal2, 1), slippage_costs, 0), k = -1)

#calculate the returns based on the price data times the signals. must add to signal a day to buy on the day
#after the signal
SPY$portfolio.return_no_costs2 <- (ROC(SPY$SPY.Adjusted)*lag(signal2))
SPY$portfolio.return_withcosts2 <- SPY$portfolio.return_no_costs2 + SPY$slippage_signal2


table.AnnualizedReturns(SPY[,c(11,12,13,16,17)])
chart.CumReturns(SPY[,c(11,12,13,16,17)], legend.loc = "topleft")
table.DrawdownsRatio(SPY[,c(11,12,13,16,17)])
maxDrawdown(SPY[,c(11,12,13,16,17)])

#Before costs the OBV as confirmation for trend was helpful from a risk and return and lower drawdowns
#compared to SMA trend alone and passive benchmark (all before costs)

#lets explore GE
head(GE)
GE$OBV <- OBV(GE$GE.Adjusted,GE$GE.Volume)
autoplot(GE$OBV)

GE$SMA <- SMA(GE$GE.Adjusted, 200)

#create the signal 
#this signal says if the price is above the trend and OBV is positive
signal <- ifelse(GE$GE.Adjusted > GE$SMA & GE$OBV > 0, 1,
                 ifelse(GE$GE.Adjusted < GE$SMA & GE$OBV < 0,-1, 0))

#add slippage and transaction costs in percentage term including slippage and commissions per Philisophical Economics on historical costs
slippage_costs <- -.0060

#add the signal and the trend strategy returns
GE$signal <- signal

#add slippage and commision costs signal
GE$slippage_signal <- lag(ifelse(GE$signal != lag(GE$signal, 1), slippage_costs, 0), k = -1)


#calculate the returns based on the price data times the signals. must add to signal a day to buy on the day
#after the signal
GE$portfolio.return_no_costs <- (ROC(GE$GE.Adjusted)*lag(signal))
GE$portfolio.return_withcosts <- GE$portfolio.return_no_costs + GE$slippage_signal
GE$Passive_Returns <- Return.calculate(GE$GE.Adjusted)

table.AnnualizedReturns(GE[,c(11:13)])
chart.CumReturns(GE[,c(11:13)], legend.loc = "topleft")

#lets see if the OBV + SMA is better than simple SMA long short
signal2 <- ifelse(GE$GE.Adjusted < GE$SMA, -1, 1)
GE$signal2 <- signal2

#add slippage and commision costs signal
GE$slippage_signal2 <- lag(ifelse(GE$signal2 != lag(GE$signal2, 1), slippage_costs, 0), k = -1)

#calculate the returns based on the price data times the signals. must add to signal a day to buy on the day
#after the signal
GE$portfolio.return_no_costs2 <- (ROC(GE$GE.Adjusted)*lag(signal2))
GE$portfolio.return_withcosts2 <- GE$portfolio.return_no_costs2 + GE$slippage_signal2


table.AnnualizedReturns(GE[,c(11,12,13,16,17)])
chart.CumReturns(GE[,c(11,12,13,16,17)], legend.loc = "topleft")
table.DrawdownsRatio(GE[,c(11,12,13,16,17)])
maxDrawdown(GE[,c(11,12,13,16,17)])

#long flat SMA
signal3 <- ifelse(GE$GE.Adjusted > GE$SMA, 1, 0)
GE$signal3 <- signal3

#add slippage and commision costs signal
GE$slippage_signal3 <- lag(ifelse(GE$signal3 != lag(GE$signal3, 1), slippage_costs, 0), k = -1)

#calculate the returns based on the price data times the signals. must add to signal a day to buy on the day
#after the signal
GE$portfolio.return_no_costs3 <- (ROC(GE$GE.Adjusted)*lag(signal3))
GE$portfolio.return_withcosts3 <- GE$portfolio.return_no_costs3 + GE$slippage_signal3


table.AnnualizedReturns(GE[,c(11,12,13,16,17,20,21)])
chart.CumReturns(GE[,c(11,12,13,16,17,20,21)], legend.loc = "topleft")
chart.CumReturns(GE[,c(11,13,16,20)], legend.loc = "topleft")
table.DrawdownsRatio(GE[,c(11,12,13,16,17,20,21)])
maxDrawdown(GE[,c(11,12,13,16,17,20,21)])

#in this instance a long short SMA would have done better which is driven by taking advantage
#of the downtrend in price, so it is apparent at quick gland that if you have a long term
#up trend bias, like with equities a long flat is better, but if it is commodoties or other assets
#which you don't have high confidence in upward bias than long short could do better


