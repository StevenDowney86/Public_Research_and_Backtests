library(corrplot)
library(tidyverse)
library(PerformanceAnalytics)
library(tibbletime)
library(tidyquant)
library(quantmod)

#This script is analyzing the performance and risk contribution for adding Bitwise 10 cryptoasset
#index to a stanard portfolio

#set the class Date for importing csv file
setClass('myDate')
setAs("character","myDate", function(from) as.Date(from, format = "%m/%d/%y"))
Bitwise10 <- read.csv(file.choose(),
                                colClasses = c('myDate',
                                               'numeric'),
                                header = TRUE)

#convert to an xts class for calculations
Bitwise10 <- na.omit(Bitwise10)
Bitwise10_xts <- as.xts(Bitwise10[,2], order.by = Bitwise10$Date)
head(Bitwise10_xts)
colnames(Bitwise10_xts) <- "Bitwise 10"

#get ETFs as asset class proxies
symbols <- c("VT","LQD","GLD","GSG","IEF","RWO","HYG","EEM")

start <- "2017-10-02"
end <- "2019-08-31"
  
getSymbols(symbols, from = start, to = end)

#merge the data and then calculate the returns while removing NAs
returns <- merge(VT[,6],
                 LQD[,6],
                 GLD[,6],
                 GSG[,6],
                 IEF[,6],
                 RWO[,6],
                 HYG[,6],
                 EEM[,6],
                 Bitwise10_xts) %>% Return.calculate() %>% na.omit()
colnames(returns) <- c("VT", "LQD", "GLD", "GSG", "IEF", "RWO", "HYG", "EEM", "Bitwise 10")

#create correlogram and correlation matrix
Asset_corr <- cor(returns)

corrplot(Asset_corr,
         method = "circle", type = "lower", order = "alphabet",
         tl.cex = .8, cl.cex = .8, number.cex = .8)

#create weights to put into portfolio rebalance function
weights_8 = rep(1/8, times = 8)
weights_9 = rep(1/9, times = 9)
weights_10 = c(rep(.96/8, times =8),.04)


standard_portfolio_equal <- Return.portfolio(returns[,c(1:8)], weights = weights_8,
                                       rebalance_on = "months")
portfolio_equal_Bitwise <- Return.portfolio(returns[,c(1:9)], weights = weights_9,
                                             rebalance_on = "months")
portfolio_equal_Bitwise_04 <- Return.portfolio(returns, weights = weights_10,
                                            rebalance_on = "months")

portfolios <- merge(standard_portfolio_equal,
      portfolio_equal_Bitwise_04,
      portfolio_equal_Bitwise)

colnames(portfolios) <- c("Equal Weight ex-Bitwise",
                          "Equal Weight ETF and 4% Bitwise",
                          "Equal Weight ETF and Bitwise")

chart.CumReturns(portfolios, legend.loc = "topleft", main = "Crypotassets and Your Portfolio")
table.AnnualizedReturns(portfolios)
table.DrawdownsRatio(portfolios)

chart.CumReturns(returns, legend.loc = "topright", main = "Cumulative performance of Bitwise 10 vs. ETFs")
SortinoRatio(portfolios)
chart.Drawdown(returns, legend.loc = "bottomleft", main = "Cumulative performance of Bitwise 10 vs. ETFs")
