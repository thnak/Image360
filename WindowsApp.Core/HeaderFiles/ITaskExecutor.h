#pragma once
#include "Types.h"

namespace WindowsApp::Core
{
    class ITaskExecutor
    {
    public:
        virtual ~ITaskExecutor() = default;

        // Contract: must be idempotent — re-invoked with the same Task
        // after a crash must be safe and produce the same committed
        // result. Expected failures (bad input, transient I/O error)
        // return false and should set an error message on the task's
        // owning ProjectManager row via the caller; exceptions are
        // reserved for programmer errors, not runtime conditions.
        virtual bool Execute(Task& task, CancellationToken token) = 0;
    };
}
