# smooth_ces.R

library(smooth)

complex_es <- function(y, h=10, level = c(0.8,0.95)){

	ces_m = list()
	ces_u = list()
	ces_l = list()
	for (l in level){
		fit <- ces(y, h = h, level = l, interval = "nonparametric")
		ces_m <- fit$forecast
		ces_u[paste0("l_",l)] <- list(fit$upper)
		ces_l[paste0("l_",l)] <- list(fit$lower)

	}
	
	return(
		list(method = "ces",
			mean = as.vector(ces_m), # take one as they are all the same
			upper = as.matrix(as.data.frame(ces_u)),
			lower = as.matrix(as.data.frame(ces_l))
		))
}
# jnk <- complex_es(df$value);jnk