using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public class Stack<T>
    {
        private T[] _stack;
        private int _stackSize;
        private int _topPointer = -1;

        public Stack(int stackSize)
        {
            _stackSize = stackSize;
            _stack = new T[_stackSize];
        }

        public int GetNumberOfItems()
        {
            return _topPointer + 1;
        }

        public T Peek()
        {
            if (_topPointer < 0)
            {
                throw new InvalidOperationException("Stack is empty");
            }
            return _stack[_topPointer];
        }

        public T Pop()
        {
            if (_topPointer < 0)
            {
                throw new InvalidOperationException("Stack is empty");
            }
            _topPointer--;
            return _stack[_topPointer + 1];
        }

        public void Push(T item)
        {
            if (_topPointer + 1 == _stackSize)
            {
                throw new InvalidOperationException("Stack is full");
            }
            _topPointer++;
            _stack[_topPointer] = item;
        }

        public bool IsEmpty()
        {
            return _topPointer == -1;
        }
    }
}
