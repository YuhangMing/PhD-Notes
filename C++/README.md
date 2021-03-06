# Random notes in C++

### std::set<> vs. std::vector<>
* std::set<>: elements are ordered, and unique, fast in insertion & deleting in the middle of the container;
* std::vector<>: elementes are unordered, and can contain duplicates, fast in inserion & deleting at the end of the container.
* [ref1-GeeksforGeeks](https://www.geeksforgeeks.org/difference-between-stdset-vs-stdvector-in-c-stl/), [ref2-StackOverflow](https://stackoverflow.com/questions/8686725/what-is-the-difference-between-stdset-and-stdvector).

### Smart Pointer (C++11)
Automatically delete the allocated memory. No longer need to delete the pointer manually. [ref-GeeksforGeeks](https://www.geeksforgeeks.org/smart-pointers-cpp/)

### Static keyword

详细解释见[link](https://zhuanlan.zhihu.com/p/37439983)

静态成员变量有以下特点：

1. 静态成员变量是该类的所有对象所共有的。对于普通成员变量，每个类对象都有自己的一份拷贝。而静态成员变量一共就一份，无论这个类的对象被定义了多少个，静态成员变量只分配一次内存，由该类的所有对象共享访问。所以，静态数据成员的值对每个对象都是一样的，它的值可以更新；
2. 因为静态数据成员在全局数据区分配内存，由本类的所有对象共享，所以，它不属于特定的类对象，不占用对象的内存，而是在所有对象之外开辟内存，在没有产生类对象时其作用域就可见。因此，在没有类的实例存在时，静态成员变量就已经存在，我们就可以操作它；
3. 静态成员变量存储在全局数据区。static 成员变量的内存空间既不是在声明类时分配，也不是在创建对象时分配，而是在初始化时分配。静态成员变量必须初始化，而且只能在类体外进行。否则，编译能通过，链接不能通过。在Example 5中，语句int Myclass::Sum=0;是定义并初始化静态成员变量。初始化时可以赋初值，也可以不赋值。如果不赋值，那么会被默认初始化，一般是 0。静态数据区的变量都有默认的初始值，而动态数据区（堆区、栈区）的变量默认是垃圾值。
4. static 成员变量和普通 static 变量一样，编译时在静态数据区分配内存，到程序结束时才释放。这就意味着，static 成员变量不随对象的创建而分配内存，也不随对象的销毁而释放内存。而普通成员变量在对象创建时分配内存，在对象销毁时释放内存。

何时采用静态数据成员？

设置静态成员（变量和函数）这种机制的目的是将某些和类紧密相关的全局变量和函数写到类里面，看上去像一个整体，易于理解和维护。如果想在同类的多个对象之间实现数据共享，又不要用全局变量，那么就可以使用静态成员变量。也即，静态数据成员主要用在各个对象都有相同的某项属性的时候。

同全局变量相比，使用静态数据成员有两个优势：
1) 静态成员变量没有进入程序的全局命名空间，因此不存在与程序中其它全局命名冲突的可能。
2) 可以实现信息隐藏。静态成员变量可以是private成员，而全局变量不能。

与静态成员变量类似，我们也可以声明一个静态成员函数。

静态成员函数为类服务而不是为某一个类的具体对象服务。静态成员函数与静态成员变量一样，都是类的内部实现，属于类定义的一部分。普通成员函数必须具体作用于某个对象，而静态成员函数并不具体作用于某个对象。

普通的成员函数一般都隐含了一个this指针，this指针指向类的对象本身，因为普通成员函数总是具体地属于类的某个具体对象的。当函数被调用时，系统会把当前对象的起始地址赋给 this 指针。通常情况下，this是缺省的。如函数fn()实际上是this->fn()。

与普通函数相比，静态成员函数属于类本身，而不作用于对象，因此它不具有this指针。正因为它没有指向某一个对象，所以它无法访问属于类对象的非静态成员变量和非静态成员函数，它只能调用其余的静态成员函数和静态成员变量。从另一个角度来看，由于静态成员函数和静态成员变量在类实例化之前就已经存在可以访问，而此时非静态成员还是不存在的，因此静态成员不能访问非静态成员。
