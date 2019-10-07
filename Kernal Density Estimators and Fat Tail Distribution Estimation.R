library(quantmod)

getSymbols(c("AXP", "CAT", "SBUX"), from="2007-01-03", to="2016-12-30", src="yahoo")
chartSeries(AXP, theme="white")
chartSeries(CAT, theme="white")
chartSeries(SBUX, theme="white")

# Compute net returns
CAT.netret = diff(CAT$CAT.Adjusted)/lag(CAT$CAT.Adjusted) 
AXP.netret = diff(AXP$AXP.Adjusted)/lag(AXP$AXP.Adjusted)
SBUX.netret = diff(SBUX$SBUX.Adjusted)/lag(SBUX$SBUX.Adjusted)

# Discriptive statistic
summary(CAT.netret)
sd(CAT.netret, na.rm=TRUE)
mad(CAT.netret, na.rm=TRUE) # a robust estimate for standard deviation

summary(AXP.netret)
sd(AXP.netret, na.rm=TRUE)
mad(AXP.netret, na.rm=TRUE) # a robust estimat for standard deviation

summary(SBUX.netret)
sd(SBUX.netret, na.rm=TRUE)
mad(SBUX.netret, na.rm=TRUE) # a robust estimat for standard deviatio

# Compute log returns
CAT.logret = diff(log(CAT$CAT.Adjusted))
AXP.logret = diff(log(AXP$AXP.Adjusted))
SBUX.logret = diff(log(SBUX$SBUX.Adjusted))
logrets = list(CAT.logret, AXP.logret, SBUX.logret)

#compute qqplot and KDE and normal distribution
labels = list("CAT", "AXP", "SBUX")
par(mfrow=c(3,2))
for (i in 1:length(logrets)) {
  retn = coredata(logrets[[i]])
  qqnorm(retn, datax = TRUE,
         xlab = "normal quantile", # this is for the theoretical quantiles
         ylab = paste("log returns of ", labels[[i]]), # this is for the sample quantiles
         main = paste("normal probability plot for log return of ", labels[[i]]))
  qqline(retn, datax=TRUE, col = 2)
  d <- density(retn, adjust = 1, na.rm = TRUE)
  m=mean(retn, na.rm=TRUE)
  stdv = mad(retn, na.rm=TRUE)
  plot(d, type = "n", xlim=c(m-3*stdv,m+3*stdv),
       main= paste("KDE for log returns of", labels[[i]], "and a normal density\nwith the same mean and variance"))
  polygon(d, col = "wheat")
  z = seq(from=m-5*stdv,to=m+5*stdv,by=stdv/100)
  lines(z,dnorm(z,mean=m, sd=stdv), lty=2,lwd=3,col="red")
}


#Comparing Equity retruns with t distribution
par(mfrow=c(3,4))
logrets = list(CAT.logret, AXP.logret, SBUX.logret)
labels = list("CAT", "AXP", "SBUX")
tdfs = c(3,2,2)
for (i in 1:length(logrets)) {
  retn = coredata(logrets[[i]])
  n = length(retn)
  grid = (1:n)/(n+1)
  for (tdf in tdfs[[i]]:(tdfs[[i]]+2)) {
    qqplot(retn, qt(grid,df=tdf),
           main= paste("t-probability plot of", labels[[i]], "\ndf =", tdf),
           xlab=paste(labels[[i]], "log-returns"),
           ylab="t-quantiles")
    qntl = quantile(retn, probs = c(0.25,0.75), na.rm=TRUE, names=FALSE)
    abline( lm( qt(c(.25,.75),df = tdf) ~ qntl ), col = 2)
  }
  d <- density(retn, adjust = 1, na.rm = TRUE)
  m=mean(retn, na.rm=TRUE)
  stdv = mad(retn, na.rm=TRUE)
  plot(d, type = "n", xlim=c(m-5*stdv,m+5*stdv),
       main= paste("KDE for log returns of", labels[[i]], "and\n a scaled t-density with DF=", tdfs[[i]]+1))
  polygon(d, col = "wheat")
  x = seq(from=m-10*stdv,to=m+10*stdv,by=stdv/100)
  lines(x,dt((x-m)/stdv,df=tdfs[[i]]+1)/stdv, lty=2,lwd=3,col="red")
}

#Which distribution has the fattest tails?

library(nimble)

n = 1000

set.seed(100)
data_rdexp = rdexp(n, location = 0, scale = 1)
data_rnorm = rnorm(n, mean = 0, sd = 1)
data_rlnorm = rlnorm(n, meanlog = 0, sdlog = 1)
data_rcauchy = rcauchy(n, location = 0, scale = 1)

qqplot(x = data_rnorm, y = data_rdexp, datax = TRUE,
       xlab = "normal quantile", # this is for the theoretical quantiles
       ylab = paste("Double Exponential quantiles"), # this is for the sample quantiles
       main = paste("normal probability plot for Double Exponential"))
qqline(data_rnorm, datax=TRUE, col = 2)

qqplot(x = data_rnorm, y = data_rlnorm, datax = TRUE,
       xlab = "normal quantile", # this is for the theoretical quantiles
       ylab = paste("Log Normal quantiles"), # this is for the sample quantiles
       main = paste("normal probability plot for Log Normal"))
qqline(data_rnorm, datax=TRUE, col = 2)

qqplot(x = data_rlnorm, y = data_rcauchy, datax = TRUE,
       xlab = "log normal quantile", # this is for the theoretical quantiles
       ylab = paste("Cauchy quantiles"), # this is for the sample quantiles
       main = paste("Log normal probability plot for Cauchy"))
qqline(data_rlnorm, datax=TRUE, col = 2)

##Calculating Sum of Squared Errors
getSymbols(c("DAAA","DFF","DGS10","DGS30"), src = "FRED")

data <- na.omit(merge(DAAA, DFF, DGS10, DGS30))

c(dim(data))

DAAA_dif = diff(as.vector(data[,"DAAA"]))
DFF_dif = diff(as.vector(data[,"DFF"]))
DGS10_dif = diff(as.vector(data[,"DGS10"]))
DGS30_dif = diff(as.vector(data[,"DGS30"]))

summary(lm(DAAA_dif ~ DGS10_dif + DGS30_dif + DFF_dif))

anova_data = anova(lm(DAAA_dif ~ DGS10_dif + DGS30_dif + DFF_dif))

regression_SS = sum(anova_data[,"Sum Sq"][1:3])
residual_error_SS = sum(anova_data[,"Sum Sq"][4])

totalSS = regression_SS + residual_error_SS

totalSS
