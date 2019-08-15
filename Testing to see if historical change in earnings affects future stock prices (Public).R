#trying to see if change in earnings trend is predictive of future stock 
#prices and 

library(PerformanceAnalytics)
library(quantmod)
library(tidyverse)
library(Quandl)
library(ggthemes)
library(gridExtra)
library(gtable)
library(grid)
library(TTR)

#create custome ggplot theme
#from http://joeystanley.com/blog/custom-themes-in-ggplot2
theme_joey <- function () { 
  theme_bw(base_size=12, base_family="Avenir") %+replace% 
    theme(
      panel.background  = element_blank(),
      plot.background = element_rect(fill="gray96", colour=NA), 
      legend.background = element_rect(fill="transparent", colour=NA),
      legend.key = element_rect(fill="transparent", colour=NA)
    )
}

#choose start and end dates
start <- as.Date("1900-01-01")
end <- Sys.Date()

#set api key
Quandl.api_key("apikeyhere")

Yale <- Quandl("YALE/SPCOMP", api_key="apikeyhere", type = "xts")


#create 12 month SMA of Earnings as a leading indicator
SP_Earnings <- na.omit(Yale$Earnings)
SP_Earnings$SMA12 <- SMA(SP_Earnings, n = 12)
SP_Earnings$Year.on.Year.Change <- ROC(SP_Earnings[,1], n = 12)

colnames(SP_Earnings) <- c("S&P 500 Earnings",
                           "12-Month Moving Average Earnings",
                           "Year on Year Change")

#calculate the price return, dividend yield, monthly equivalent, and total return
SP_Earnings$Change_in_trend <- Return.calculate(SP_Earnings$`Year on Year Change`)

SP500 <- merge(Yale$`S&P Composite`, Yale$Dividend)

SP500$Price_Return <- Return.calculate(SP500$S.P.Composite)

SP500$Dividend_Yield <- SP500$Dividend/SP500$S.P.Composite
SP500$Dividend_Yield_Monthly <- SP500$Dividend_Yield/12

SP500$Total_Return <- SP500$Price_Return + SP500$Dividend_Yield_Monthly

SP500 <- na.omit(SP500)

#check to make sure it looks correct
head(SP500)
table.AnnualizedReturns(SP500[,c(3,5,6)])

#create a date time index that will work with other functions in 
#performance analytics package and match length of current
#data from Yale Shiller. I chose 28th of the month since it 
#will be at the end of the month and don't have to worry about February

#dates <- seq(as.Date("1871-02-28"),length=1778,by="months")

#SP500_xts <- cbind(SP500$Total_Return, dates)

#SP500_data <- as.xts(SP500_xts$Total_Return, order.by = dates)

#colnames(SP500_data) <- "SP500 TR"

#SP500_data$rolling_12month_returns <- rollapply(SP500_data[,1],
                                      #FUN = Return.annualized,
                                      #width = 12,
                                      #scale = 12)

#SP500_data$rolling_24month_returns <- rollapply(SP500_data[,1],
                                                #FUN = Return.annualized,
                                                #width = 24,
                                                #scale = 12)

#SP500_data$rolling_36month_returns <- rollapply(SP500_data[,1],
                                                #FUN = Return.annualized,
                                                #width = 36,
                                                #scale = 12)
#check to make sure the data looks correct
#autoplot(SP500_data[,c(2,3,4)], facets = FALSE)

#lag the total return SP500 performance to correlate with the current month 
#earnings and the next 12, 24, 36 month returns
#SP500_data$rolling_12month_returns_lag <- lag.xts(SP500_data$rolling_12month_returns, k = -12)
#SP500_data$rolling_24month_returns_lag <- lag.xts(SP500_data$rolling_24month_returns, k = -24)
#SP500_data$rolling_36month_returns_lag <- lag.xts(SP500_data$rolling_36month_returns, k = -36)

head(SP500)

SP500$rolling_12month_returns <- rollapply(SP500$Total_Return,
                                                FUN = Return.annualized,
                                                width = 12,
                                                scale = 12)

SP500$rolling_24month_returns <- rollapply(SP500$Total_Return,
                                                FUN = Return.annualized,
                                                width = 24,
                                                scale = 12)

SP500$rolling_36month_returns <- rollapply(SP500$Total_Return,
                                                FUN = Return.annualized,
                                                width = 36,
                                                scale = 12)

#lag the total return SP500 performance to correlate with the current month 
#earnings and the next 12, 24, 36 month returns
SP500$rolling_12month_returns_lag <- lag.xts(SP500$rolling_12month_returns, k = -12)
SP500$rolling_24month_returns_lag <- lag.xts(SP500$rolling_24month_returns, k = -24)
SP500$rolling_36month_returns_lag <- lag.xts(SP500$rolling_36month_returns, k = -36)

tail(SP500)

Test_data <- merge(SP500, SP_Earnings)
Test_data_na <- na.omit(Test_data)

tail(Test_data_na, 5)


#the chart of correlation seems to indicate that change in earnings on a calendar basis
#is not correlated with future returns
chart.Correlation(Test_data_na[,c(10,11,12,15,16)])

#create a trading strategy to see if getting in and out depending on earnings
#can work

#1st signal and strategy is if the change in trend is negative get out
signal <- ifelse(Test_data_na$Change_in_trend > 0, 1, 0)

#add the signal 
Test_data_na$signal <- signal

#calculate the returns based on the price data times the signals. must add to signal three months to buy on the day
#after the signal
Test_data_na$signal_lagged <- lag(signal,3)
Test_data_na$portfolio.return_no_costs_1 <- (Test_data_na$Total_Return*Test_data_na$signal_lagged)
tail(Test_data_na, 30)
ncol(Test_data_na)
table.AnnualizedReturns(Test_data_na[,c(6,19)])

#second signal is to see if change in trend is greater than -.05 otherwise get out
signal2 <- ifelse(Test_data_na$Change_in_trend > -.05, 1, 0)

#add the signal 
Test_data_na$signal2 <- signal2

#calculate the returns based on the price data times the signal. must add to signal three months to buy on the day
#after the signal
Test_data_na$signal_lagged2 <- lag(signal2,3)
Test_data_na$portfolio.return_no_costs_2 <- (Test_data_na$Total_Return*Test_data_na$signal_lagged2)
tail(Test_data_na, 30)
table.AnnualizedReturns(Test_data_na[,c(6,22)])

#Third signal to see if simple year on year change is greater than -.05 otherwise get out
signal3 <- ifelse(Test_data_na$Year.on.Year.Change > -.05, 1, 0)

#add the signal 
Test_data_na$signal3 <- signal3

#calculate the returns based on the price data times the signal. must add to signal three months to buy on the day
#after the signal
Test_data_na$signal_lagged3 <- lag(signal3,3)
Test_data_na$portfolio.return_no_costs_3 <- (Test_data_na$Total_Return*Test_data_na$signal_lagged3)
tail(Test_data_na, 30)
table.AnnualizedReturns(Test_data_na[,c(6,25)])

chart.CumReturns(Test_data_na[,c(6,19,22,25)], legend.loc = "topleft")

#based on a few different signals, even ignoring transaction costs, one
#cannot use earnings, change in earnings, or change in earnings trend
#to time the market and get superior risk adjusted performance
