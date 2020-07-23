# This code originates from the M4 Competition Statistical Benchmarks.
# https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R

library(forecast) #Requires v8.2

naive_seasonal <- function(input, fh){
  #Used to estimate Seasonal Naive
  frcy <- frequency(input)
  frcst <- naive(input, h=fh)$mean 
  if (frcy>1){ 
    frcst <- head(rep(as.numeric(tail(input,frcy)), fh), fh) + frcst - frcst
  }
  return(frcst)
}

#Used to determine whether a time series is seasonal
SeasonalityTest <- function(input, ppy){
    tcrit <- 1.645
    if (length(input)<3*ppy){
        test_seasonal <- FALSE
    }else{
        xacf <- acf(input, plot = FALSE)$acf[-1, 1, 1]
        clim <- tcrit/sqrt(length(input)) * sqrt(cumsum(c(1, 2 * xacf^2)))
        test_seasonal <- ( abs(xacf[ppy]) > clim[ppy] )
        
        if (is.na(test_seasonal)==TRUE){ test_seasonal <- FALSE }
    }
    
    return(test_seasonal)
}

#Used to estimate the statistical benchmarks of the M4 competition
Benchmarks <- function(input, fh){
    ppy <- frequency(input)
    ST <- F
    if (ppy>1){ ST <- SeasonalityTest(input,ppy) }
    if (ST==T){
        Dec <- decompose(input,type="multiplicative")
        des_input <- input/Dec$seasonal
        len <- length(Dec$seasonal)
        SIout <- head(rep(Dec$seasonal[(len-ppy+1):len], fh), fh)
    }else{
        des_input <- input
        SIout <- rep(1, fh)
    }
    
    f1 <- naive(input, h=fh)$mean #Naive
    f2 <- naive_seasonal(input, fh=fh) #Seasonal Naive
    f3 <- naive(des_input, h=fh)$mean*SIout #Naive2
    f4 <- ses(des_input, h=fh)$mean*SIout #Ses
    f5 <- holt(des_input, h=fh, damped=F)$mean*SIout #Holt
    f6 <- holt(des_input, h=fh, damped=T)$mean*SIout #Damped
    f8 <- (f4+f5+f6)/3 #Comb
    
    return(list(f1,f2,f3,f4,f5,f6,f8))
}
