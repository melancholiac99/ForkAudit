import benchmark as B
B.CONFIGS['async_graph']=dict(B.CONFIGS['graph'],async_scheduling=True)
if __name__=='__main__':B.main('async_graph')
