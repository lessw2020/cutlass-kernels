// 4 stage pipeline
// a learning example for using cutlass PipelineAsync;
static constexpr int NumStages = 4;

using MainloopPipeline = typename cutlass::PipelineAsync<NumStages>;
using PipelineState = typename cutlass::PipelineState<NumStages>;

// 1 producer threads and 2 consumer threads
typename MainloopPipeline::Params params;
params.producer_arv_count = 1;
params.consumer_arv_count = 2;

MainloopPipeline pipeline(shared_storage.storage, params);

//Producer threads
if (thread_idx ==0) {
    PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
    for (; iter>0; --iter) {
        pipeline.producer_acquire(smem_pipe_write);

        // Producer ops go here

        pipeline.producer_commit(smem_pipe_write);
        ++smem_pipe_write;
    }
}
else if (thread_idx==1 or thread_idx==2) {
    PipelineState smem_pipe_read;
    for (; iter >0; --iter) {
        pipeline.consumer_wait(smem_pipe_read);

        //Consumer ops
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
    }
}
