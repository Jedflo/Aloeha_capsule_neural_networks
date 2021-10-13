import pandas

def logFile_update(log_file_dir):
    log_file = pandas.read_csv(log_file_dir)
    row_count = log_file.shape[0]

    
    index = 0
    while index < row_count:
        log_file.loc[index, "epoch"] = index
        index+=1
        progress = round((index/row_count)*100)
        print('Rearranging log file:', progress, end='\r')
    print('Rearranging log file: Done!')
    log_file.to_csv(log_file_dir, index=False)
