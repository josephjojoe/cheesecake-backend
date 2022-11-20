using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public class Queue<T>
    {
        private T[] queue;
        private int headPointer = 0;
        private int tailPointer = -1;
        private int _queueSize;

        public Queue(int queueSize)
        {
            queue = new T[queueSize];
            _queueSize = queueSize;
        }

        public T Peek()
        {
            if (headPointer > tailPointer)
            {
                throw new InvalidOperationException("Queue is empty");
            }
            return queue[headPointer];
        }

        public T PeekEnd()
        {
            if (headPointer > tailPointer)
            {
                throw new InvalidOperationException("Queue is empty");
            }
            return queue[tailPointer];
        }

        public void Enqueue(T item)
        {
            if (tailPointer + 1 == _queueSize)
            {
                throw new InvalidOperationException("Queue is full");
            }
            else
            {
                tailPointer++;
                queue[tailPointer] = item;
            }
        }

        public T Dequeue()
        {
            if (headPointer > tailPointer)
            {
                throw new InvalidOperationException("Queue is empty");
            }
            else
            {
                headPointer++;
                return queue[headPointer - 1];
            }
        }

        public int GetQueueSize()
        {
            return _queueSize;
        }
    }
}
