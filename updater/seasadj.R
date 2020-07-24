# This code originates from the M4 Competition Statistical Benchmarks.
# https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R

#  f1 <- naive(input, h=fh)$mean #Naive
#  f2 <- naive_seasonal(input, fh=fh) #Seasonal Naive
#  f3 <- naive(des_input, h=fh)$mean*SIout #Naive2
#  f4 <- ses(des_input, h=fh)$mean*SIout #Ses
#  f5 <- holt(des_input, h=fh, damped=F)$mean*SIout #Holt
#  f6 <- holt(des_input, h=fh, damped=T)$mean*SIout #Damped
#  f7 <- Theta.classic(input=des_input, fh=fh)$mean*SIout #Theta
#  f8 <- (f4+f5+f6)/3 #Comb

library(forecast) #Requires v8.2

# Some example data
# p <- presidents
# p[ is.na(p) ] <- 0.0
# SeasonalityTest( p, 4 )

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
naive2 <- function(y, h = 10, level = c(80,95)){
    ppy <- frequency(y)
    ST <- (ppy>1) && SeasonalityTest(y,ppy)
    if (ST){
        Dec <- decompose(y,type=type)
        seasadj <- y/Dec$seasonal
        SIout <- head(rep(Dec$seasonal[1:ppy], h), h)
    }else{
        seasadj <- y
        SIout <- rep(1, h)
    }
    fc <- naive(seasadj, h=h, level=level)
    return( list( method = "Seasonally Adjusted Naive",
                  mean = fc$mean*SIout,
                  upper = fc$upper*SIout,
                  lower = fc$lower*SIout) )
}
