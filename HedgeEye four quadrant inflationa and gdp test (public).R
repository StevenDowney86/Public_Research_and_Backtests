#HedgeEye is a service that forecasts and recommends asset allocations
#based on whether GDP is accelerating/decelerating and Inflation is 
#accelerating/decelerating. By that the mean not only is GDP positive but 
#the rate of change is increasing (accelerating) or the rate of change
#is declining (decelerating)

#Four Quadrants
#Stagflation (current GDP decelerating and current CPI accelerating)
#High Inflation Expansion (current GDP accelerting and current CPI accelerating)
#Low Inflation Expansion (current GDP accelerating and current CPI decelerating)
#Recession (current GDP decelerating and current CPI decelerating)

#By definition, a slowing growth for GDP would precede
#a recession and a slowing of a contraction of GDP would precede an expansion.

#1.	What is the primary question I am trying to answer?
  
#The question is whether sectioning history into those four quandrants shows a 
#meaningfully and statistically difference in returns and risk and whether 
#you can implement a systematic strategy to capitalize on that, in light of the
#reporting time lag for GDP (quarterly) and CPI (monthly)

#2.	What is the hypothesis that I want to test, and potentially refute 
#if I am thinking from a null hypothesis orientation?

#If assets are put into the four quadrants, will there returns and risk
#be statistically different than their average over the full time horizon?

#If the null hypothesis that asset returns aren't statistically different in 
#different environments can be rejected, the real question is can it be implemented
#systematically in real-time.

#3.Has someone else already conducted similar research, whether a white paper 
#or peer reviewed journal article?
  
#There is research about how real GDP growth and subsequent stock returns 
#do not have a strong correlation, with a hypothesis that all GDP growth 
#doesn't filter down to publicly traded company earnings and that future
#high growth is priced in and equity prices are already rise before hand

#https://www.msci.com/documents/10199/a134c5d5-dca0-420d-875d-06adb948f578

#http://pages.stern.nyu.edu/~dbackus/GE_asset_pricing/PiazzesiSchneider%20inflation%20Oct%2005.pdf
#the above paper looks at the high inflation 1970's and how increased inflation 
#expectations (and realized high inflation) would lead to lower corporate profitability
#and thuse lower stock returns

#https://investresolve.com/inc/uploads/pdf/ReSolve-Inside-Our-DNA.pdf
#page 11 has their view on how different assets do well in different economic
#environments correlating to my research objective

#From their paper:
#Note that, intuitively, bonds react favorably
#to disinflation and deflation, while commodities can be expected to
#deliver strong returns during periods of accelerating inflation. It also
#makes sense that stocks would flourish during periods of economic
#growth, while cash, gold and long-duration fixed income assets
#excel when the economy is shrinking.


#4.What are other firms working on and who are our competitors?

#Resolve Asset Management, Bridgewater Risk Parity, AQR

#5. Are there any fundamental or economic theories/reasons behind why my 
#hypothesis should or should not work?

#Different Asset Classes are driven by different drivers, with stocks enjoying
#high returns with low inflation, strong economic growth, and gov't bonds
#doing well with economic contraction and low inflation, and commodities/gold expected
#to do well with high inflation as real assets.

#Since the market is forward-looking the real challenge will be if you can create
#a timing model in advance of the market moves.

#6.nAre there any behavioral reasons/theories behind why my hypothesis 
#should or should not work?
  
#Not that I can think of

#7. Gather the Data -try to get the data that has the longest history, 
#across geographies, and asset classes

#I have:
  #AQR commodities equal weight index back to 1890 from AQR
  #USA Stocks TR and 10 Year Treasury Bond back to 1835 from GFD
  #Gold (USD) back to 1970 via Quandl - London Gold
  #Housing Data from Shiller back to 1890 from Yale through Quandl
  #Quarterly Real GDP for USA from 1947 from St. Louis FRED
  #Monthly Inflation USD from 1875 from GFD

#so data from 1947 is all we have for pieces of data
  

library(PerformanceAnalytics)
library(tidyverse)
library(quantmod)
library(Quandl)
library(RColorBrewer)
library(ggthemes)
library(gridExtra)
library(gtable)
library(grid)

Quandl.api_key(api_key = "MFiHkkmpYSxDhfZ1ygrU")

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

#Get GDP, Inflation, and Housing Data
getSymbols("A191RL1Q225SBEA", src = "FRED")
GDP <- A191RL1Q225SBEA

getSymbols("CPALTT01USQ659N", src = "FRED")
Inflation <- CPALTT01USQ659N

#import GFD Inflation Data if want to use
#load("~/Coding/R/Working Directory/USA Inflation 1789 - 2019.RData")

    #since monthly starts at 1875 filter data
    #Inflation_USA_1875 <- Inflation_USA_xts["1875/"]

getSymbols("MICH", src = "FRED")
Inflation_Expectations_UMich <- MICH

#Using this index from Yale/Shiller since goes back further than NAREIT index (1971)
Nominal_Housing <- Quandl("YALE/NHPI", type = "xts")
colnames(Nominal_Housing) <- "Nominal Housing Index"

Gold <- Quandl("LBMA/GOLD", api_key="apikeyhere", start_date="1966-01-01", end_date=Sys.time(), type = "xts")

#8. Clean the Data and make sure it is formatted in a easy way to analysis
  #I will need to convert it to quarterly index data and then convert to returns
  #since GDP is only quarterly

#import Asset Class Returns

load("~/Coding/R/Working Directory/Total Return Indices/Composite Data/USA Tbills, Bonds, and Stocks and Recession Dates 1790 - mid 2018.RData")
load("~/Coding/R/Working Directory/Total Return Indices/Composite Data/All Total Returns Indices.RData")
load("~/Coding/R/Working Directory/Total Return Indices/AQR Commodities 1877 - June 2018.RData")
#take monthly returns and convert to an index or take monthly index and convert to quarterly
#then calculate returns
Quarterly_Asset_Index <- to.quarterly(USA_10yr_Bond_xts$`USA 10 yr. T-Bonds Nominal TR Index`, OHLC = FALSE)
Quarterly_Asset_Index$USA_Stock_TR_Index <- to.quarterly(USA_Stocks_xts$`USA Equity Nominal TR Index`, OHLC = FALSE)

Commodities <- cumprod(1+ na.omit(Commodities_AQR_xts$Excess.return.of.equal.weight.commodities.portfolio))
Quarterly_Asset_Index$Commodities <- to.quarterly(Commodities, OHLC = FALSE)

tail(Quarterly_Asset_Index)

Gold <- Gold[,1] %>% na.omit()
Quarterly_Asset_Index$Gold <- to.quarterly(Gold, OHLC = FALSE)

Quarterly_Asset_Index$Housing <- to.quarterly(Nominal_Housing, OHLC = FALSE)


colnames(Quarterly_Asset_Index) <- c("US 10 Yr T-Bonds",
                                     "US Stocks",
                                     "Eq Wght Commodities",
                                     "Gold",
                                     "Housing")

Quarterly_Asset_Returns <- Return.calculate(Quarterly_Asset_Index)

#Create In Sample and Out of Sample Data

Insample_data_dates <- "1960/1997"
Outofsample_data_dates <- "1998/2018-06"

#9. In Sample Research

#how to measure rate of change and acceleration or deceleration for GDP and CPI?

#can't use simple Rate of Change calculation due to shifts when GDP is negative
#creates NaN 
#data$ROC_GDP <- ROC(data$GDP, n = 1)

#you can take the absolute value but even then the data looks messy
#data$ROC_GDP <- ROC(abs(data$GDP), n = 1)

#create the trend of GDP and Inflation, which help manage the negative number
#issue and show if GDP/Inflation is higher/lower than trend
#Created a trend blend mixing 1,2,3,4 quater SMA into an average as an ensemble approach
data <- merge(GDP, Inflation) %>% na.omit()
colnames(data) <- c("GDP","CPI")

Trend <- SMA(data$GDP, n = 1)
Trend$SMA2 <- SMA(data$GDP, n = 2)
Trend$SMA3 <- SMA(data$GDP, n = 3)
Trend$SMA4 <- SMA(data$GDP, n = 4)

Average_Trend <- as.xts(apply(Trend, FUN = mean, MARGIN = 1))
tail(Average_Trend)

colnames(Average_Trend) <- "Average GDP Trend"

tail(Average_Trend)

CPI_Trend <- SMA(data$CPI, n = 1)
CPI_Trend$SMA2 <- SMA(data$CPI, n = 2)
CPI_Trend$SMA3 <- SMA(data$CPI, n = 3)
CPI_Trend$SMA4 <- SMA(data$CPI, n = 4)

Average_Trend_CPI <- as.xts(apply(CPI_Trend, FUN = mean, MARGIN = 1))
tail(Average_Trend_CPI)

colnames(Average_Trend_CPI) <- "Average CPI Trend"

tail(Average_Trend_CPI)

Average_Trends <- merge(Average_Trend,Average_Trend_CPI)
head(Average_Trends)

#having a hard time merging the actual data and trend data so taking out
#coredata and creating a date index and then creating fresh xts
Trend_data <- coredata(Average_Trends)
data_updated <- coredata(data)

data_merged <- cbind(Trend_data,data_updated)

dates <- seq(as.Date("1960-01-01"),length=237,by="quarters")
data_updated <- as.xts(x = data_merged, order.by = dates)
head(data_updated)

#plot the actual data vs. average trend to make sure looks correct
autoplot(data_updated[,c(1,3)], facets = NULL)
autoplot(data_updated[,c(2,4)], facets = NULL)

#below I was trying to convert the quarterly data to monthly but decided against it as converting
#monthly returns to quartlery was easier

#dates <- seq(as.Date("1960-01-01"),length=702,by="months")
#head(dates)
#tail(dates)
#xts2 <- xts(x=US_BONDS_1790_to_present_returns["1960/"], order.by=dates)
#xts3 <- merge(xts2, data)
#tail(xts3)
#xts4 <- na.locf(xts3, fromLast = TRUE)

#convert the trend data to quarterly xts format for easy merge
data_updated_qtr <- to.quarterly(data_updated, OHLC = FALSE)
tail(data_updated_qtr)
<<<<<<< HEAD

Quarterly_Asset_Returns_GDPCPI_test1 <- merge(data_updated_qtr, Quarterly_Asset_Returns)
Quarterly_Asset_Returns_GDPCPI_test1 <- Quarterly_Asset_Returns_GDPCPI_test1[Insample_data_dates] %>% na.omit()

head(Quarterly_Asset_Returns_GDPCPI_test1)

#Explore the Data

#current GDP equal to or above trend and current CPI equal or above trend
Expansion_returns <- Quarterly_Asset_Returns_GDPCPI_test1[,c(5:ncol(Quarterly_Asset_Returns_GDPCPI_test1))][Quarterly_Asset_Returns_GDPCPI_test1$Average.GDP.Trend <= Quarterly_Asset_Returns_GDPCPI_test1$GDP & Quarterly_Asset_Returns_GDPCPI_test1$Average.CPI.Trend <= Quarterly_Asset_Returns_GDPCPI_test1$CPI]

#current GDP below trend and current CPI above trend
Stagflation_returns <- Quarterly_Asset_Returns_GDPCPI_test1[,c(5:ncol(Quarterly_Asset_Returns_GDPCPI_test1))][Quarterly_Asset_Returns_GDPCPI_test1$Average.GDP.Trend >= Quarterly_Asset_Returns_GDPCPI_test1$GDP & Quarterly_Asset_Returns_GDPCPI_test1$Average.CPI.Trend <= Quarterly_Asset_Returns_GDPCPI_test1$CPI]

#current GDP above trend and current CPI below trend
Goldilocks_returns <- Quarterly_Asset_Returns_GDPCPI_test1[,c(5:ncol(Quarterly_Asset_Returns_GDPCPI_test1))][Quarterly_Asset_Returns_GDPCPI_test1$Average.GDP.Trend <= Quarterly_Asset_Returns_GDPCPI_test1$GDP & Quarterly_Asset_Returns_GDPCPI_test1$Average.CPI.Trend >= Quarterly_Asset_Returns_GDPCPI_test1$CPI]

#current GDP below trend and current CPI below trend
Recession_returns <- Quarterly_Asset_Returns_GDPCPI_test1[,c(5:ncol(Quarterly_Asset_Returns_GDPCPI_test1))][Quarterly_Asset_Returns_GDPCPI_test1$Average.GDP.Trend >= Quarterly_Asset_Returns_GDPCPI_test1$GDP & Quarterly_Asset_Returns_GDPCPI_test1$Average.CPI.Trend >= Quarterly_Asset_Returns_GDPCPI_test1$CPI]

#entire period
Insample_period_returns <- Quarterly_Asset_Returns_GDPCPI_test1[,c(5:ncol(Quarterly_Asset_Returns_GDPCPI_test1))]

#take the quartlery return data, create a cumulative index of returns, and then annualize
Goldilocks_Annaulized_Returns <- cumprod(1+Goldilocks_returns)
Goldilocks_Annualized_Returns <- tail(Goldilocks_Annaulized_Returns,1)^(1/((nrow(Goldilocks_Annaulized_Returns)/4)))-1

Recession_Annaulized_Returns <- cumprod(1+Recession_returns)
Recession_Annualized_Returns <- tail(Recession_Annaulized_Returns,1)^(1/((nrow(Recession_Annaulized_Returns)/4)))-1

Stagflation_Annaulized_Returns <- cumprod(1+Stagflation_returns)
Stagflation_Annualized_Returns <- tail(Stagflation_Annaulized_Returns,1)^(1/((nrow(Stagflation_Annaulized_Returns)/4)))-1

Expansion_Annaulized_Returns <- cumprod(1+Expansion_returns)
Expansion_Annualized_Returns <- tail(Expansion_Annaulized_Returns,1)^(1/((nrow(Expansion_Annaulized_Returns)/4)))-1

Insample_Annualized_Returns <- cumprod(1+Insample_period_returns)
Insample_Annualized_Returns <- tail(Insample_Annualized_Returns,1)^(1/((nrow(Insample_Annualized_Returns)/4)))-1
Insample_Annualized_Returns

#check to make sure previous data corresponds to the function annualized returns
table.AnnualizedReturns(Insample_period_returns)
#looks good

Recession_df <- gather(as.data.frame(Recession_Annualized_Returns))
Expansion_df <- gather(as.data.frame(Expansion_Annualized_Returns))
Goldilocks_df <- gather(as.data.frame(Goldilocks_Annualized_Returns))
Stagflation_df <- gather(as.data.frame(Stagflation_Annualized_Returns))
total_period <- gather(as.data.frame(Insample_Annualized_Returns))

Stagflation_plot <- ggplot(data=Stagflation_df, aes(x=key, y=value)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_joey()+
  labs(title ="current GDP below trend and current CPI above trend", 
       x = "Asset Class", y = "Annualized Return", caption = "Source: AQR, Quandl, St. Louis Fed")

Recession_plot <- ggplot(data=Recession_df, aes(x=key, y=value)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_joey()+
  labs(title ="current GDP below trend and current CPI below trend", 
       x = "Asset Class", y = "Annualized Return")

Expansion_plot <- ggplot(data=Expansion_df, aes(x=key, y=value)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_joey()+
  labs(title ="current GDP above trend and current CPI above trend", 
       x = "Asset Class", y = "Annualized Return")

Goldilocks_plot <- ggplot(data=Stagflation_df, aes(x=key, y=value)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_joey()+
  labs(title ="current GDP above trend and current CPI below trend", 
       x = "Asset Class", y = "Annualized Return")

grid.arrange(
  Goldilocks_plot,
  Expansion_plot,
  Recession_plot,
  Stagflation_plot,
  nrow = 2, ncol = 2,
  top = "Annualized Returns for Different Regimes 1968-1997"
)

Insample_returns_plot <- ggplot(data=total_period, aes(x=key, y=value)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_joey()+
  labs(title ="Total Period (1968-1997)", 
       x = "Asset Class", y = "Annualized Return")
Insample_returns_plot

#trying to figure out why the total returns over sample period seem higher for stocks
#than the composite of the four different economic sections?

#answer is that the function annualized return table would calculate it annualized but not
#factor in that data sets may only have 1-2 quarters in each year but they would count that as
#the annual. So I had to manually created cumprod index and then take to the power of number of years
#to get the annualized return for each time period.

#Well this 1-4 quarter blended trend model is to noisy and there isn't a long enough lag
#to give a strong contrasting signal and there is a lot of short term switching

table.AnnualizedReturns(Insample_period_returns)
chart.Drawdown(Insample_period_returns$US.Stocks)
autoplot(data_updated[,c(1,3)]["1970/1976"], facets = NULL)

#Let's look at a simple 4 quater (1 year trend lag)

test2 <- data

test2$GDP_SMA4 <- SMA(test2$GDP, n = 4)
test2$CPI_SMA4 <- SMA(test2$CPI, n = 4)

test2 <- merge(test2,Quarterly_Asset_Returns) %>% na.omit()

test2 <- test2[Insample_data_dates]

#current GDP equal to or above trend and current CPI equal or above trend
Expansion_returns2 <- test2[,c(5:ncol(test2))][test2$GDP_SMA4 <= test2$GDP & test2$CPI_SMA4 <= test2$CPI]

#current GDP below trend and current CPI above trend
Stagflation_returns2 <- test2[,c(5:ncol(test2))][test2$GDP_SMA4 >= test2$GDP & test2$CPI_SMA4 <= test2$CPI]

#current GDP above trend and current CPI below trend
Goldilocks_returns2 <- test2[,c(5:ncol(test2))][test2$GDP_SMA4 <= test2$GDP & test2$CPI_SMA4 >= test2$CPI]

#current GDP below trend and current CPI below trend
Recession_returns2 <- test2[,c(5:ncol(test2))][test2$GDP_SMA4 >= test2$GDP & test2$CPI_SMA4 >= test2$CPI]

#take the quartlery return data, create a cumulative index of returns, and then annualize
Goldilocks_Annaulized_Returns2 <- cumprod(1+Goldilocks_returns2)
Goldilocks_Annualized_Returns2 <- tail(Goldilocks_Annaulized_Returns2,1)^(1/((nrow(Goldilocks_Annaulized_Returns2)/4)))-1

Recession_Annaulized_Returns2 <- cumprod(1+Recession_returns2)
Recession_Annualized_Returns2 <- tail(Recession_Annaulized_Returns2,1)^(1/((nrow(Recession_Annaulized_Returns2)/4)))-1

Stagflation_Annaulized_Returns2 <- cumprod(1+Stagflation_returns2)
Stagflation_Annualized_Returns2 <- tail(Stagflation_Annaulized_Returns2,1)^(1/((nrow(Stagflation_Annaulized_Returns2)/4)))-1

Expansion_Annaulized_Returns2 <- cumprod(1+Expansion_returns2)
Expansion_Annualized_Returns2 <- tail(Expansion_Annaulized_Returns2,1)^(1/((nrow(Expansion_Annaulized_Returns2)/4)))-1


Recession_df2 <- gather(as.data.frame(Recession_Annualized_Returns2))
Expansion_df2 <- gather(as.data.frame(Expansion_Annualized_Returns2))
Goldilocks_df2 <- gather(as.data.frame(Goldilocks_Annualized_Returns2))
Stagflation_df2 <- gather(as.data.frame(Stagflation_Annualized_Returns2))


Stagflation_plot2 <- ggplot(data=Stagflation_df2, aes(x=key, y=value)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_joey()+
  labs(title ="current GDP below trend and current CPI above trend", 
       x = "Asset Class", y = "Annualized Return", caption = "Source: AQR, Quandl, St. Louis Fed")

Recession_plot2 <- ggplot(data=Recession_df2, aes(x=key, y=value)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_joey()+
  labs(title ="current GDP below trend and current CPI below trend", 
       x = "Asset Class", y = "Annualized Return")

Expansion_plot2 <- ggplot(data=Expansion_df2, aes(x=key, y=value)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_joey()+
  labs(title ="current GDP above trend and current CPI above trend", 
       x = "Asset Class", y = "Annualized Return")

Goldilocks_plot2 <- ggplot(data=Stagflation_df2, aes(x=key, y=value)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_joey()+
  labs(title ="current GDP above trend and current CPI below trend", 
       x = "Asset Class", y = "Annualized Return")

grid.arrange(
  Goldilocks_plot2,
  Expansion_plot2,
  Recession_plot2,
  Stagflation_plot2,
  nrow = 2, ncol = 2,
  top = "Annualized Returns for Different Regimes 1968-1997"
)

Insample_returns_plot <- ggplot(data=total_period, aes(x=key, y=value)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_joey()+
  labs(title ="Total Period (1968-1997)", 
       x = "Asset Class", y = "Annualized Return")
Insample_returns_plot

#by definition this signal is more smooth and less whipsaw
table.AnnualizedReturns(Insample_period_returns)
chart.Drawdown(Insample_period_returns$US.Stocks)
autoplot(test2[,c(1,3)]["1970/1976"], facets = NULL)

#From 1968 - 1997 high inflation from the  1960's-1970's seems to a be big factor
#affecting returns and the previous sections showing the highest stock returns come from
#declining GDP and declinging CPI is mainly driven by declining CPI, but the below correlation
#matrix shows that there is only a strong correlation between housing-CPI and no other economic
#and financial variable. My hunch is that the out of sample will look very different than the 
#in sample data.

chart.Correlation(test2, method = "spearman")

#Now let us look if we can time any of thise with a systematic model

#buy Stocks if GDP is above trend
#buy gold if CPI above trend or GDP is below trend
#buy Commodities if CPI above trend or GDP is above trend
#buy housing if CPI is above trend or GDP is above trend
#buy T-bonds if CPI is below trend or GDP is below trend


#Need to lag the data by a quarter for actual purchasing ability and will use simple annual
#trend model vs. blended 1-4 quarter trend

lagged_Asset_data <- lag.xts(Quarterly_Asset_Returns, -1)
lagged_data <- merge(test2[,c(1:4)],lagged_Asset_data) %>% na.omit()
head(lagged_data)

Stock_portfolio <- lagged_data[,c(1:4,6)]
Stock_portfolio$signal <- ifelse(Stock_portfolio$GDP >= Stock_portfolio$GDP_SMA4,1,0)
Stock_portfolio$returns <- Stock_portfolio$US.Stocks*Stock_portfolio$signal
chart.CumReturns(Stock_portfolio[,c(5,7)], legend.loc = "topleft")
table.AnnualizedReturns(Stock_portfolio[,c(5,7)])
table.DrawdownsRatio(Stock_portfolio[,c(5,7)])

summary(lm(Stock_portfolio$US.Stocks ~  Stock_portfolio$returns))
#so this simple timing with stocks did far worse then a buy and hold on an absolute basis
#but did have less volatility. There is statistical significance of a buy and hold outperforming
#the timing strategy

#buy gold if CPI above trend or GDP is below trend

Gold_portfolio <- lagged_data[,c(1:4,8)]
Gold_portfolio$signal <- ifelse(Gold_portfolio$GDP <= Gold_portfolio$GDP_SMA4 | Gold_portfolio$CPI >= Gold_portfolio$CPI_SMA4,1,0)
Gold_portfolio$returns <- Gold_portfolio$Gold*Gold_portfolio$signal
chart.CumReturns(Gold_portfolio[,c(5,7)], legend.loc = "topleft")
table.AnnualizedReturns(Gold_portfolio[,c(5,7)])
table.DrawdownsRatio(Gold_portfolio[,c(5,7)])

summary(lm(Gold_portfolio$returns ~ Gold_portfolio$Gold))
#this strategy actually did better then the static buy and hold, but there isn't statistical
#significance above a p value of 0.05


#buy Commodities if CPI above trend or GDP is above trend
Commodities_portfolio <- lagged_data[,c(1:4,7)]
Commodities_portfolio$signal <- ifelse(Commodities_portfolio$GDP >= Commodities_portfolio$GDP_SMA4 | Commodities_portfolio$CPI >= Commodities_portfolio$CPI_SMA4,1,0)
Commodities_portfolio$returns <- Commodities_portfolio$Eq.Wght.Commodities*Commodities_portfolio$signal
chart.CumReturns(Commodities_portfolio[,c(5,7)], legend.loc = "topleft")
table.AnnualizedReturns(Commodities_portfolio[,c(5,7)])
table.DrawdownsRatio(Commodities_portfolio[,c(5,7)])

summary(lm(Commodities_portfolio$Eq.Wght.Commodities ~ Commodities_portfolio$returns))

#The buy and hold did better, but primarily from the 1988-1997 period, there is no statisical
#significance that either buy and hold or the strategy generated alpha over the other but the buy
#and hold did better on returns and return/risk ratios


#buy housing if CPI is above trend or GDP is above trend
Housing_portfolio <- lagged_data[,c(1:4,9)]
Housing_portfolio$signal <- ifelse(Housing_portfolio$GDP >= Housing_portfolio$GDP_SMA4 | Housing_portfolio$CPI >= Housing_portfolio$CPI_SMA4,1,0)
Housing_portfolio$returns <- Housing_portfolio$Housing*Housing_portfolio$signal
chart.CumReturns(Housing_portfolio[,c(5,7)], legend.loc = "topleft")
table.AnnualizedReturns(Housing_portfolio[,c(5,7)])
table.DrawdownsRatio(Housing_portfolio[,c(5,7)])

summary(lm(Housing_portfolio$Housing ~ Housing_portfolio$returns))

#buy and hold housing did much better than the transition strategy, but also in reality
#buying and switching between physical housing ins't possible. During this period housing mainly
#went straight up

#buy T-bonds if CPI is below trend or GDP is below trend

Bond_portfolio <- lagged_data[,c(1:4,5)]
Bond_portfolio$signal <- ifelse(Bond_portfolio$GDP <= Bond_portfolio$GDP_SMA4 | Bond_portfolio$CPI <= Bond_portfolio$CPI_SMA4,1,0)
Bond_portfolio$returns <- Bond_portfolio$US.10.Yr.T.Bonds*Bond_portfolio$signal
chart.CumReturns(Bond_portfolio[,c(5,7)], legend.loc = "topleft")
chart.CumReturns(Bond_portfolio[,c(5,7)]["/1984"], legend.loc = "topleft")
table.AnnualizedReturns(Bond_portfolio[,c(5,7)])
table.DrawdownsRatio(Bond_portfolio[,c(5,7)])

summary(lm(Bond_portfolio$US.10.Yr.T.Bonds ~ Bond_portfolio$returns))

#bonds went up for most of the time even though real returns would have declined with high
#inflation. The buy and hold did better on returns and most return/risk ratios but there is no
#statistical difference between the two strategeis with p value of .16 and t stat 1.38

#What if we blend all 5 timing portfolios vs an equal weight buy and hold for all 5 asset
#classes. Though I doubt it will improve.

Timing_portfolio <- merge(Stock_portfolio$returns,
                          Gold_portfolio$returns,
                          Commodities_portfolio$returns,
                          Housing_portfolio$returns,
                          Bond_portfolio$returns)

Timing_portfolio$combined <- Return.portfolio(Timing_portfolio,
                                              weights = c(.2,.2,.2,.2,.2),
                                              rebalance_on = "quarters")
head(Timing_portfolio)


Buy_and_hold_portfolio <- merge(Stock_portfolio$US.Stocks,
                          Gold_portfolio$Gold,
                          Commodities_portfolio$Eq.Wght.Commodities,
                          Housing_portfolio$Housing,
                          Bond_portfolio$US.10.Yr.T.Bonds)

Buy_and_hold_portfolio$combined <- Return.portfolio(Buy_and_hold_portfolio,
                                              weights = c(.2,.2,.2,.2,.2),
                                              rebalance_on = "quarters")
head(Buy_and_hold_portfolio)

table.AnnualizedReturns(Buy_and_hold_portfolio)
table.AnnualizedReturns(Timing_portfolio)

#the buy hold combo portfoio had a higher return and return/risk ratio, but the one asset that seems
#to skew everything is physical housing with its insanely high 2.4 Sharpe Ratio.

#Need to examine how the portfolio would have done with REITs vs. housing for next time.


#a.	Review the normal testing errors and make sure I didn’t commit them:
  #i.	Look Ahead Bias - for this data set the econ data will have a publishing lag
        #Based on the Alfred St. Louis Fed database that gives when actual data was published
        #GDP data usually has a two month lag and CPI has a 1 - 2 month lag. 
        #so to be conservative and keep the data manageable I will lag the GDP and CPI
        #data by 1 quarter. So the Q1 CPI/GDP data will be actionable at the end of Q2/beginning of Q3
    #ii.	Data Snooping - I did get the idea of HedgeEye but it didn't seem to work anyways
  #iii.	Idea only working on a single market/time frame
  #iv.	Ignoring realistic transaction costs, slippage, market movement - 
      #this would be a challenge with housing but the other markets you could use futures or currently ETFs with
      #with minimal transaction costs
  #v.	Parameter Fragility – (i.e. with a trend following system if it breaks down with slightly faster/slower time frames then it probably won’t work out of sample)
      #this could be an issue with using the 12 month (4 quarter) signal

#10. Out of Sample Testing - though I am skeptical of finding anything significant even before
#time lag factorered in

#a.	Analyze the out of sample data based on my research hypothesis and findings
tail(Quarterly_Asset_Returns)
Quarterly_Asset_Returns_OOS <- lag.xts(Quarterly_Asset_Returns, -1)
Quarterly_Asset_Returns_OOS <- Quarterly_Asset_Returns_OOS[Outofsample_data_dates]

test3 <- data

test3$GDP_SMA4 <- SMA(test3$GDP, n = 4)
test3$CPI_SMA4 <- SMA(test3$CPI, n = 4)

test3 <- merge(test3,Quarterly_Asset_Returns_OOS) %>% na.omit()

test3 <- test3[Outofsample_data_dates]
head(test3)


Stock_portfolio_OOS <- test3[,c(1:4,6)]
Stock_portfolio_OOS$signal <- ifelse(Stock_portfolio_OOS$GDP >= Stock_portfolio_OOS$GDP_SMA4,1,0)
Stock_portfolio_OOS$returns <- Stock_portfolio_OOS$US.Stocks*Stock_portfolio_OOS$signal
chart.CumReturns(Stock_portfolio_OOS[,c(5,7)], legend.loc = "topleft")
table.AnnualizedReturns(Stock_portfolio_OOS[,c(5,7)])
table.DrawdownsRatio(Stock_portfolio_OOS[,c(5,7)])

summary(lm(Stock_portfolio_OOS$US.Stocks ~  Stock_portfolio_OOS$returns))
#so this simple timing with stocks did far worse then a buy and hold on an absolute basis
#but did have less volatility. There is statistical significance of a buy and hold outperforming
#the timing strategy

#buy gold if CPI above trend or GDP is below trend

Gold_portfolio_OOS <- test3[,c(1:4,8)]
Gold_portfolio_OOS$signal <- ifelse(Gold_portfolio_OOS$GDP <= Gold_portfolio_OOS$GDP_SMA4 | Gold_portfolio_OOS$CPI >= Gold_portfolio_OOS$CPI_SMA4,1,0)
Gold_portfolio_OOS$returns <- Gold_portfolio_OOS$Gold*Gold_portfolio_OOS$signal
chart.CumReturns(Gold_portfolio_OOS[,c(5,7)], legend.loc = "topleft")
table.AnnualizedReturns(Gold_portfolio_OOS[,c(5,7)])
table.DrawdownsRatio(Gold_portfolio_OOS[,c(5,7)])

summary(lm(Gold_portfolio_OOS$returns ~ Gold_portfolio_OOS$Gold))
#this strategy had better drawdown ratio stats but lower returns though higher Sharpe Ratio
#buy and hold did statistically better than timing portfolio.

#buy Commodities if CPI above trend or GDP is above trend
Commodities_portfolio_OOS <- test3[,c(1:4,7)]
Commodities_portfolio_OOS$signal <- ifelse(Commodities_portfolio_OOS$GDP >= Commodities_portfolio_OOS$GDP_SMA4 | Commodities_portfolio_OOS$CPI >= Commodities_portfolio_OOS$CPI_SMA4,1,0)
Commodities_portfolio_OOS$returns <- Commodities_portfolio_OOS$Eq.Wght.Commodities*Commodities_portfolio_OOS$signal
chart.CumReturns(Commodities_portfolio_OOS[,c(5,7)], legend.loc = "topleft")
table.AnnualizedReturns(Commodities_portfolio_OOS[,c(5,7)])
table.DrawdownsRatio(Commodities_portfolio_OOS[,c(5,7)])

summary(lm(Commodities_portfolio_OOS$Eq.Wght.Commodities ~ Commodities_portfolio_OOS$returns))

#no outperformance for the commodities timing model in any way with inferior return/risk 

#buy housing if CPI is above trend or GDP is above trend
Housing_portfolio_OOS <- test3[,c(1:4,9)]
Housing_portfolio_OOS$signal <- ifelse(Housing_portfolio_OOS$GDP >= Housing_portfolio_OOS$GDP_SMA4 | Housing_portfolio_OOS$CPI >= Housing_portfolio_OOS$CPI_SMA4,1,0)
Housing_portfolio_OOS$returns <- Housing_portfolio_OOS$Housing*Housing_portfolio_OOS$signal
chart.CumReturns(Housing_portfolio_OOS[,c(5,7)], legend.loc = "topleft")
table.AnnualizedReturns(Housing_portfolio_OOS[,c(5,7)])
table.DrawdownsRatio(Housing_portfolio_OOS[,c(5,7)])
maxDrawdown(Housing_portfolio_OOS[,c(5,7)])

summary(lm(Housing_portfolio_OOS$Housing ~ Housing_portfolio_OOS$returns))

#buy and hold housing did much better than the transition strategy, but also in reality
#buying and switching between physical housing ins't possible. Even with boom and bust of housing
#the timing model didn't help

#buy T-bonds if CPI is below trend or GDP is below trend

Bond_portfolio_OOS <- test3[,c(1:4,5)]
Bond_portfolio_OOS$signal <- ifelse(Bond_portfolio_OOS$GDP <= Bond_portfolio_OOS$GDP_SMA4 | Bond_portfolio_OOS$CPI <= Bond_portfolio_OOS$CPI_SMA4,1,0)
Bond_portfolio_OOS$returns <- Bond_portfolio_OOS$US.10.Yr.T.Bonds*Bond_portfolio_OOS$signal
chart.CumReturns(Bond_portfolio_OOS[,c(5,7)], legend.loc = "topleft")
table.AnnualizedReturns(Bond_portfolio_OOS[,c(5,7)])
table.DrawdownsRatio(Bond_portfolio_OOS[,c(5,7)])

summary(lm(Bond_portfolio_OOS$US.10.Yr.T.Bonds ~ Bond_portfolio_OOS$returns))

#once again hard to time the bond market based off of inflation and GDP data. buy and hold signficantly
#out performanced in all respects.


#b.	Exam test statistics
#i.	what are the relevant test statistics for risk factor attribution?
  #ii.	What are the proper test statistics that will reveal statistical significance?
  #t stat and p value of timing model vs. buy and hold. All failed and all return/risk ratios were inferior
#c.	If possible, test on other market histories (i.e. if examined US sector equities look at Japan sector equities)

#In summary, the model didn't work due to there not being a strong linear relationship between CPI and GDP
#and returns of all 5 asset classes when a general buy and hold did better.

