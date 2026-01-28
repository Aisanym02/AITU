// Practical Work №5
// Реализация параллельных структур данных на GPU (CUDA)
//
// В данной работе я реализую параллельный стек и очередь
// с использованием атомарных операций CUDA.
// Цель — обеспечить безопасный доступ нескольких потоков
// и сравнить производительность реализованных структур.
// Структура параллельного стека (LIFO)
struct Stack {
    int* data;        // массив данных в глобальной памяти
    int top;          // индекс вершины стека
    int capacity;     // максимальная емкость стека

    // Инициализация стека
    __device__ void init(int* buffer, int size) {
        data = buffer;
        top = -1;
        capacity = size;
    }

    // Добавление элемента в стек (push)
    // Использую atomicAdd для безопасного доступа из нескольких потоков
    __device__ bool push(int value) {
        int pos = atomicAdd(&top, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        // Если стек переполнен — откатываю top
        atomicSub(&top, 1);
        return false;
    }

    // Извлечение элемента из стека (pop)
    __device__ bool pop(int* value) {
        int pos = atomicSub(&top, 1);
        if (pos >= 0) {
            *value = data[pos];
            return true;
        }
        // Если стек пуст — возвращаю значение top обратно
        atomicAdd(&top, 1);
        return false;
    }
};
// Структура параллельной очереди (FIFO)
struct Queue {
    int* data;        // массив данных в глобальной памяти
    int head;         // индекс чтения
    int tail;         // индекс записи
    int capacity;     // максимальная емкость очереди

    // Инициализация очереди
    __device__ void init(int* buffer, int size) {
        data = buffer;
        head = 0;
        tail = 0;
        capacity = size;
    }

    // Добавление элемента в очередь (enqueue)
    // Использую atomicAdd для потокобезопасной записи
    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        // Очередь переполнена — откатываю tail
        atomicSub(&tail, 1);
        return false;
    }

    // Извлечение элемента из очереди (dequeue)
    __device__ bool dequeue(int* value) {
        int pos = atomicAdd(&head, 1);
        if (pos < tail) {
            *value = data[pos];
            return true;
        }
        // Очередь пуста — откатываю head
        atomicSub(&head, 1);
        return false;
    }
};
// Ядро CUDA для параллельного тестирования стека
// Каждый поток выполняет операции push и pop
__global__ void stack_kernel(int* buffer, int capacity,
                             int* pop_result,
                             int* push_ok, int* pop_ok) {
    __shared__ Stack stack;

    // Инициализацию выполняет один поток
    if (threadIdx.x == 0)
        stack.init(buffer, capacity);

    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Параллельный push
    if (stack.push(tid))
        atomicAdd(push_ok, 1);

    __syncthreads();

    // Параллельный pop
    int value;
    if (stack.pop(&value)) {
        pop_result[tid] = value;
        atomicAdd(pop_ok, 1);
    }
}
// После выполнения ядра я проверяю:
// 1) количество успешных операций
// 2) корректность работы стека и очереди
// 3) среднее время выполнения
// Это позволяет сравнить производительность параллельных структур данных.
