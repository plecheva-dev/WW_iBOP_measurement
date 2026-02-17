Scripts use functions from the custom package, load data, and 
produce results that are exported into the data folder.

It should contain: 

```python 
import ...

der main():
    ...


if __name__ == "__main__":
    main()
    
```
Scripts should **always** depend on packages!
Packages should **never** depend on scripts !