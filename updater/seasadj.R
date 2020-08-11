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

library( "forecast" ) #Requires v8.2

# Check the acf to see whether any seasonality is present.
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

# Python passes y as a vector for which frequency(y) == 1.
# Try to reconstruct the frequency from the timestamps.
# Adapted from
# https://stackoverflow.com/questions/19217729/check-the-frequency-of-time-series-data
guess_period <- function(x) { 
  average_period <- as.double( mean(diff(x)), units="days" )
  difference <- abs(log( average_period / c( 1, 7/5, 30, 91, 365 ) ) )
  freq <- c( 7, 5, 12, 4, 1 )
  freq[ which.min( difference ) ]
}

# The Naive2 statistical benchmark of the M4 competition
# 
naive2 <- function(y, h = 10, level = c(80,95)) {

    ppy <- guess_period( as.Date( names(y) ) )
    #print( paste( "ppy:", ppy ) )

    ST <- (ppy>1) && SeasonalityTest(y,ppy)
    #print( paste( "ST:", ST ) )

    if (ST){
        y <- ts(y, frequency=ppy)
        Dec <- decompose(y, type="multiplicative")
        seasadj <- y/Dec$seasonal
        SIout <- head(rep(Dec$seasonal[1:ppy], h), h)
    }else{
        seasadj <- y
        SIout <- rep(1, h)
    }
    fc <- naive(seasadj, h=h, level=level)
    return( list( method = "Seasonally Adjusted Naive",
                  seasadj = ST,
                  mean = fc$mean*SIout,
                  upper = fc$upper*SIout,
                  lower = fc$lower*SIout) )
}

# Comb benchmark from M4 competition (SES, Holt, Damped)

comb <- function(y, h=10, level = c(80,95)) {

	fses_m <- ses(y, h=h, level=level)$mean
	fses_u <- ses(y, h=h, level=level)$upper
	fses_l <- ses(y, h=h, level=level)$lower
	
	fholt_m <- holt(y, h=h, level=level, damped=F)$mean
	fholt_u <- holt(y, h=h, level=level, damped=F)$upper
	fholt_l <- holt(y, h=h, level=level, damped=F)$lower
	
	fdamp_m <- holt(y, h=h, level=level, damped=T)$mean
	fdamp_u <- holt(y, h=h, level=level, damped=T)$upper
	fdamp_l <- holt(y, h=h, level=level, damped=T)$lower
	
	return( list( method = "Comb",
		mean = (fses_m+fholt_m+fdamp_m) / 3,
		upper = (fses_u+fholt_u+fdamp_u) / 3,
		lower = (fses_l+fholt_l+fdamp_l) / 3 ))
}