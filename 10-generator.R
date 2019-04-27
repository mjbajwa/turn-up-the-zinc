# Generator Function ------------------------------------------------------

# The generator function creates data on-the-fly to feed the keras models

generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = TRUE, batch_size, step, pred = FALSE) {
  
  if (is.null(max_index)){
    max_index <- nrow(data) - delay #- 1
  }
  
  i <- min_index + lookback
  
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index){
        rows <- c(i:min(i+batch_size, max_index))
        i <<- min_index + lookback # Reset i when i + batch_size exceeds max_index
      }else{
        rows <- c(i:min(i+batch_size, max_index))
        i <<- i + length(rows)
      }
    }
    
    samples <- array(0, dim = c(length(rows),
                                (lookback+1)/step,
                                dim(data)[[-1]])) # Change this when building a NARX model.
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]],
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices] #-grep("recovery", colnames(data))] # remove prediction column.
      targets[[j]] <- data[rows[[j]] + delay, grep("recovery", colnames(data))] # Adjust to capture the right well_id
    }
    
    if(pred == TRUE){
      list(samples)
    }else{
      list(samples, targets)
    }
  }
}
