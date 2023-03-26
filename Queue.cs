using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public class Queue<T>
    {
        private T[] _queue;
        private int _headPointer = 0;
        private int _tailPointer = -1;
        private int _queueSize;

        public Queue(int queueSize)
        {
            _queue = new T[queueSize];
            _queueSize = queueSize;
        }

        public T Peek()
        {
            if (_headPointer > _tailPointer)
            {
                throw new InvalidOperationException("Queue is empty");
            }
            return _queue[_headPointer];
        }

        public T PeekEnd()
        {
            if (_headPointer > _tailPointer)
            {
                throw new InvalidOperationException("Queue is empty");
            }
            return _queue[_tailPointer];
        }

        public void Enqueue(T item)
        {
            if (_tailPointer + 1 == _queueSize)
            {
                throw new InvalidOperationException("Queue is full");
            }
            else
            {
                _tailPointer++;
                _queue[_tailPointer] = item;
            }
        }

        public T Dequeue()
        {
            if (_headPointer > _tailPointer)
            {
                throw new InvalidOperationException("Queue is empty");
            }
            else
            {
                _headPointer++;
                return _queue[_headPointer - 1];
            }
        }

        public int GetQueueSize()
        {
            return _queueSize;
        }

        public void ResetPointers()
        {
            _headPointer = 0;
            _tailPointer = -1;
        }

        public void ResetHeadPointer()
        {
            _headPointer = 0;
        }
    }
}
