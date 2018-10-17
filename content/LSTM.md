## LSTM Cells

It is comprised of 4 gates with 2 inputs and 2 outputs:

**Learn Gate:** it takes the **Short term memory** and the **Event** and combines them with a ```than``` function and then ignores a part of it by multiplying it by an _ignore factor_ i<sub>t</sub>. To calculate the _ignore factor_ it combines de Shor term memory with the event multiplies it by the _Ignore Weights_ and activates using a ```sigmoid``` function and outputs the **new Short Term Memory**.

**Forget Gate:** it takes the **Long term memory** and multiplies it by a _forget factor_ f<sub>t</sub> to calculate the _forget factor_ it combines de Shor term memory with the event multiplies it by the _Forget Weights_ and activates using a ```sigmoid``` function.

**Remember Gate:** it takes the **Forget Gate** output and adds it to the **Learn Gate** outputs the **new Long Term Memory**.

**Use Gate**: or output gate, will take the **Forget Gate** and activate it with ```tanh``` then Take the **Shor Term Memory** and activate it with ```sigmoid``` and then multiplies them, and that is the output.