ρΙ
Τͺ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ϊ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ρ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Α
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-0-g3f878cff5b68Μͺ

model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_namemodel/dense/kernel
z
&model/dense/kernel/Read/ReadVariableOpReadVariableOpmodel/dense/kernel*
_output_shapes
:	*
dtype0
x
model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namemodel/dense/bias
q
$model/dense/bias/Read/ReadVariableOpReadVariableOpmodel/dense/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:@*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
}
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:*
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
}
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*(
_output_shapes
:*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
}
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*(
_output_shapes
:*
dtype0
s
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
l
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
ζ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*‘
valueB B
ω
FeatureExtraction
AdaptiveAvgPool

Prediction
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures*
±
output_channel
ConvNet
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
ͺ
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
20
21*

"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
/11
012
113
414
515
16
17*
* 
°
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 

;serving_default* 
* 

<layer_with_weights-0
<layer-0
=layer-1
>layer-2
?layer_with_weights-1
?layer-3
@layer-4
Alayer-5
Blayer_with_weights-2
Blayer-6
Clayer-7
Dlayer_with_weights-3
Dlayer-8
Elayer-9
Flayer-10
Glayer_with_weights-4
Glayer-11
Hlayer_with_weights-5
Hlayer-12
Ilayer-13
Jlayer_with_weights-6
Jlayer-14
Klayer_with_weights-7
Klayer-15
Llayer-16
Mlayer-17
Nlayer_with_weights-8
Nlayer-18
Olayer-19
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses*

"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519*
z
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
/11
012
113
414
515*
* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
XR
VARIABLE_VALUEmodel/dense/kernel,Prediction/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEmodel/dense/bias*Prediction/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
MG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbatch_normalization/gamma&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbatch_normalization/beta'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEbatch_normalization/moving_mean'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#batch_normalization/moving_variance'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_5/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_1/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_1/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_6/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_6/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
 
-0
.1
22
33*

0
1
2*
* 
* 
* 
* 
¦

"kernel
#bias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses*

k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses* 

q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses* 
¦

$kernel
%bias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses*

}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬

&kernel
'bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬

(kernel
)bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses* 

‘	variables
’trainable_variables
£regularization_losses
€	keras_api
₯__call__
+¦&call_and_return_all_conditional_losses* 
’

*kernel
§	variables
¨trainable_variables
©regularization_losses
ͺ	keras_api
«__call__
+¬&call_and_return_all_conditional_losses*
ά
	­axis
	+gamma
,beta
-moving_mean
.moving_variance
?	variables
―trainable_variables
°regularization_losses
±	keras_api
²__call__
+³&call_and_return_all_conditional_losses*

΄	variables
΅trainable_variables
Άregularization_losses
·	keras_api
Έ__call__
+Ή&call_and_return_all_conditional_losses* 
’

/kernel
Ί	variables
»trainable_variables
Όregularization_losses
½	keras_api
Ύ__call__
+Ώ&call_and_return_all_conditional_losses*
ά
	ΐaxis
	0gamma
1beta
2moving_mean
3moving_variance
Α	variables
Βtrainable_variables
Γregularization_losses
Δ	keras_api
Ε__call__
+Ζ&call_and_return_all_conditional_losses*

Η	variables
Θtrainable_variables
Ιregularization_losses
Κ	keras_api
Λ__call__
+Μ&call_and_return_all_conditional_losses* 

Ν	variables
Ξtrainable_variables
Οregularization_losses
Π	keras_api
Ρ__call__
+?&call_and_return_all_conditional_losses* 
¬

4kernel
5bias
Σ	variables
Τtrainable_variables
Υregularization_losses
Φ	keras_api
Χ__call__
+Ψ&call_and_return_all_conditional_losses*

Ω	variables
Ϊtrainable_variables
Ϋregularization_losses
ά	keras_api
έ__call__
+ή&call_and_return_all_conditional_losses* 

"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519*
z
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
/11
012
113
414
515*
* 

ίnon_trainable_variables
ΰlayers
αmetrics
 βlayer_regularization_losses
γlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*
* 
* 
 
-0
.1
22
33*

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

"0
#1*

"0
#1*
* 

δnon_trainable_variables
εlayers
ζmetrics
 ηlayer_regularization_losses
θlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ιnon_trainable_variables
κlayers
λmetrics
 μlayer_regularization_losses
νlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

ξnon_trainable_variables
οlayers
πmetrics
 ρlayer_regularization_losses
ςlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 
* 
* 

$0
%1*

$0
%1*
* 

σnon_trainable_variables
τlayers
υmetrics
 φlayer_regularization_losses
χlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ψnon_trainable_variables
ωlayers
ϊmetrics
 ϋlayer_regularization_losses
όlayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

ύnon_trainable_variables
ώlayers
?metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

&0
'1*

&0
'1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

(0
)1*

(0
)1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
‘	variables
’trainable_variables
£regularization_losses
₯__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 
* 
* 

*0*

*0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses*
* 
* 
* 
 
+0
,1
-2
.3*

+0
,1*
* 

 non_trainable_variables
‘layers
’metrics
 £layer_regularization_losses
€layer_metrics
?	variables
―trainable_variables
°regularization_losses
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

₯non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
΄	variables
΅trainable_variables
Άregularization_losses
Έ__call__
+Ή&call_and_return_all_conditional_losses
'Ή"call_and_return_conditional_losses* 
* 
* 

/0*

/0*
* 

ͺnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
?layer_metrics
Ί	variables
»trainable_variables
Όregularization_losses
Ύ__call__
+Ώ&call_and_return_all_conditional_losses
'Ώ"call_and_return_conditional_losses*
* 
* 
* 
 
00
11
22
33*

00
11*
* 

―non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
Α	variables
Βtrainable_variables
Γregularization_losses
Ε__call__
+Ζ&call_and_return_all_conditional_losses
'Ζ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

΄non_trainable_variables
΅layers
Άmetrics
 ·layer_regularization_losses
Έlayer_metrics
Η	variables
Θtrainable_variables
Ιregularization_losses
Λ__call__
+Μ&call_and_return_all_conditional_losses
'Μ"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ήnon_trainable_variables
Ίlayers
»metrics
 Όlayer_regularization_losses
½layer_metrics
Ν	variables
Ξtrainable_variables
Οregularization_losses
Ρ__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

40
51*

40
51*
* 

Ύnon_trainable_variables
Ώlayers
ΐmetrics
 Αlayer_regularization_losses
Βlayer_metrics
Σ	variables
Τtrainable_variables
Υregularization_losses
Χ__call__
+Ψ&call_and_return_all_conditional_losses
'Ψ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Γnon_trainable_variables
Δlayers
Εmetrics
 Ζlayer_regularization_losses
Ηlayer_metrics
Ω	variables
Ϊtrainable_variables
Ϋregularization_losses
έ__call__
+ή&call_and_return_all_conditional_losses
'ή"call_and_return_conditional_losses* 
* 
* 
 
-0
.1
22
33*

<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
N18
O19*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

-0
.1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

20
31*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

serving_default_args_0Placeholder*/
_output_shapes
:????????? d*
dtype0*$
shape:????????? d
y
serving_default_args_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
α
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_5/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_6/kernelconv2d_6/biasmodel/dense/kernelmodel/dense/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_12613666
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Π	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&model/dense/kernel/Read/ReadVariableOp$model/dense/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOpConst*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_save_12614688

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemodel/dense/kernelmodel/dense/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_5/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_6/kernelconv2d_6/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__traced_restore_12614764ψγ
Ο

+__inference_conv2d_4_layer_call_fn_12614394

inputs#
unknown:
identity’StatefulPartitionedCallΪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_12612027x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ή
t
X__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_12613026

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0inputs*
T0*Ά
_output_shapes£
 :?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_splitο
stackPacksplit:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7split:output:8split:output:9split:output:10split:output:11split:output:12split:output:13split:output:14split:output:15split:output:16split:output:17split:output:18split:output:19split:output:20split:output:21split:output:22split:output:23*
N*
T0*4
_output_shapes"
 :?????????*

axisS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0stack:output:0*
T0*4
_output_shapes"
 :?????????*
	num_splity
stack_1Packsplit_1:output:0*
N*
T0*8
_output_shapes&
$:"?????????*

axisg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      z
MeanMeanstack_1:output:0Mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Έ
Μ
-__inference_sequential_layer_call_fn_12614038

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	&

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	
identity’StatefulPartitionedCallά
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_12612099x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? d
 
_user_specified_nameinputs
ν

)__inference_conv2d_layer_call_fn_12614250

inputs!
unknown:@
	unknown_0:@
identity’StatefulPartitionedCallδ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? d@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_12611935w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? d@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? d: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? d
 
_user_specified_nameinputs
Τ

&__inference_signature_wrapper_12613666

args_0

args_1!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	&

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:
identity’StatefulPartitionedCallΧ
StatefulPartitionedCallStatefulPartitionedCallargs_0args_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_12611742s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:????????? d:?????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? d
 
_user_specified_nameargs_0:OK
'
_output_shapes
:?????????
 
_user_specified_nameargs_1
έ

(__inference_model_layer_call_fn_12613334
x
text!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	&

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:
identity’StatefulPartitionedCallμ
StatefulPartitionedCallStatefulPartitionedCallxtextunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_12613187s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:????????? d:?????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:????????? d

_user_specified_nameX:MI
'
_output_shapes
:?????????

_user_specified_nametext

g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_12611751

inputs
identity’
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
°


F__inference_conv2d_1_layer_call_and_return_conditional_losses_12611959

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????2w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????2@
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12611775

inputs
identity’
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
φn
Έ
H__inference_sequential_layer_call_and_return_conditional_losses_12614241

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@B
'conv2d_1_conv2d_readvariableop_resource:@7
(conv2d_1_biasadd_readvariableop_resource:	C
'conv2d_2_conv2d_readvariableop_resource:7
(conv2d_2_biasadd_readvariableop_resource:	C
'conv2d_3_conv2d_readvariableop_resource:7
(conv2d_3_biasadd_readvariableop_resource:	C
'conv2d_4_conv2d_readvariableop_resource::
+batch_normalization_readvariableop_resource:	<
-batch_normalization_readvariableop_1_resource:	K
<batch_normalization_fusedbatchnormv3_readvariableop_resource:	M
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_5_conv2d_readvariableop_resource:<
-batch_normalization_1_readvariableop_resource:	>
/batch_normalization_1_readvariableop_1_resource:	M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_6_conv2d_readvariableop_resource:7
(conv2d_6_biasadd_readvariableop_resource:	
identity’"batch_normalization/AssignNewValue’$batch_normalization/AssignNewValue_1’3batch_normalization/FusedBatchNormV3/ReadVariableOp’5batch_normalization/FusedBatchNormV3/ReadVariableOp_1’"batch_normalization/ReadVariableOp’$batch_normalization/ReadVariableOp_1’$batch_normalization_1/AssignNewValue’&batch_normalization_1/AssignNewValue_1’5batch_normalization_1/FusedBatchNormV3/ReadVariableOp’7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1’$batch_normalization_1/ReadVariableOp’&batch_normalization_1/ReadVariableOp_1’conv2d/BiasAdd/ReadVariableOp’conv2d/Conv2D/ReadVariableOp’conv2d_1/BiasAdd/ReadVariableOp’conv2d_1/Conv2D/ReadVariableOp’conv2d_2/BiasAdd/ReadVariableOp’conv2d_2/Conv2D/ReadVariableOp’conv2d_3/BiasAdd/ReadVariableOp’conv2d_3/Conv2D/ReadVariableOp’conv2d_4/Conv2D/ReadVariableOp’conv2d_5/Conv2D/ReadVariableOp’conv2d_6/BiasAdd/ReadVariableOp’conv2d_6/Conv2D/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0§
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@*
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@e

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? d@§
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:?????????2@*
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Δ
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2*
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2j
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2¬
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ζ
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????j
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ΐ
conv2d_3/Conv2DConv2Dre_lu_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????j
re_lu_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????¬
max_pooling2d_2/MaxPoolMaxPoolre_lu_3/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides

conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ζ
conv2d_4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:*
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:*
dtype0­
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0±
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ώ
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0y
re_lu_4/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ΐ
conv2d_5/Conv2DConv2Dre_lu_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0΅
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ι
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0{
re_lu_5/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????¬
max_pooling2d_3/MaxPoolMaxPoolre_lu_5/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides

conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Η
conv2d_6/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????j
re_lu_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????r
IdentityIdentityre_lu_6/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????λ
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? d
 
_user_specified_nameinputs
ψ
£
+__inference_conv2d_2_layer_call_fn_12614328

inputs#
unknown:
	unknown_0:	
identity’StatefulPartitionedCallη
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_12611983x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ν
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_12614309

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????2c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2:X T
0
_output_shapes
:?????????2
 
_user_specified_nameinputs
Σ
Ω
9__inference_vgg__feature_extractor_layer_call_fn_12612669
input_1!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	&

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	
identity’StatefulPartitionedCallι
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612626x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:????????? d
!
_user_specified_name	input_1
Ο

+__inference_conv2d_5_layer_call_fn_12614480

inputs#
unknown:
identity’StatefulPartitionedCallΪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_12612054x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
¨
Ή
F__inference_conv2d_5_layer_call_and_return_conditional_losses_12614487

inputs:
conv2d_readvariableop_resource:
identity’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
	
Χ
8__inference_batch_normalization_1_layer_call_fn_12614513

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12611895
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
γ
£
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12613914
xJ
0sequential_conv2d_conv2d_readvariableop_resource:@?
1sequential_conv2d_biasadd_readvariableop_resource:@M
2sequential_conv2d_1_conv2d_readvariableop_resource:@B
3sequential_conv2d_1_biasadd_readvariableop_resource:	N
2sequential_conv2d_2_conv2d_readvariableop_resource:B
3sequential_conv2d_2_biasadd_readvariableop_resource:	N
2sequential_conv2d_3_conv2d_readvariableop_resource:B
3sequential_conv2d_3_biasadd_readvariableop_resource:	N
2sequential_conv2d_4_conv2d_readvariableop_resource:E
6sequential_batch_normalization_readvariableop_resource:	G
8sequential_batch_normalization_readvariableop_1_resource:	V
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:	X
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	N
2sequential_conv2d_5_conv2d_readvariableop_resource:G
8sequential_batch_normalization_1_readvariableop_resource:	I
:sequential_batch_normalization_1_readvariableop_1_resource:	X
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	Z
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	N
2sequential_conv2d_6_conv2d_readvariableop_resource:B
3sequential_conv2d_6_biasadd_readvariableop_resource:	
identity’-sequential/batch_normalization/AssignNewValue’/sequential/batch_normalization/AssignNewValue_1’>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp’@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1’-sequential/batch_normalization/ReadVariableOp’/sequential/batch_normalization/ReadVariableOp_1’/sequential/batch_normalization_1/AssignNewValue’1sequential/batch_normalization_1/AssignNewValue_1’@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp’Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1’/sequential/batch_normalization_1/ReadVariableOp’1sequential/batch_normalization_1/ReadVariableOp_1’(sequential/conv2d/BiasAdd/ReadVariableOp’'sequential/conv2d/Conv2D/ReadVariableOp’*sequential/conv2d_1/BiasAdd/ReadVariableOp’)sequential/conv2d_1/Conv2D/ReadVariableOp’*sequential/conv2d_2/BiasAdd/ReadVariableOp’)sequential/conv2d_2/Conv2D/ReadVariableOp’*sequential/conv2d_3/BiasAdd/ReadVariableOp’)sequential/conv2d_3/Conv2D/ReadVariableOp’)sequential/conv2d_4/Conv2D/ReadVariableOp’)sequential/conv2d_5/Conv2D/ReadVariableOp’*sequential/conv2d_6/BiasAdd/ReadVariableOp’)sequential/conv2d_6/Conv2D/ReadVariableOp 
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Έ
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@*
paddingSAME*
strides

(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@{
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? d@½
 sequential/max_pooling2d/MaxPoolMaxPool#sequential/re_lu/Relu:activations:0*/
_output_shapes
:?????????2@*
ksize
*
paddingVALID*
strides
₯
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ε
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2*
paddingSAME*
strides

*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ί
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2Β
"sequential/max_pooling2d_1/MaxPoolMaxPool%sequential/re_lu_1/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
¦
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0η
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ί
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
sequential/re_lu_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????¦
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0α
sequential/conv2d_3/Conv2DConv2D%sequential/re_lu_2/Relu:activations:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

*sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ί
sequential/conv2d_3/BiasAddBiasAdd#sequential/conv2d_3/Conv2D:output:02sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
sequential/re_lu_3/ReluRelu$sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Β
"sequential/max_pooling2d_2/MaxPoolMaxPool%sequential/re_lu_3/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
¦
)sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0η
sequential/conv2d_4/Conv2DConv2D+sequential/max_pooling2d_2/MaxPool:output:01sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
‘
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes	
:*
dtype0₯
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes	
:*
dtype0Γ
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Η
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3#sequential/conv2d_4/Conv2D:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<¬
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ά
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
sequential/re_lu_4/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????¦
)sequential/conv2d_5/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0α
sequential/conv2d_5/Conv2DConv2D%sequential/re_lu_4/Relu:activations:01sequential/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
₯
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0©
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0Η
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Λ
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#sequential/conv2d_5/Conv2D:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<΄
/sequential/batch_normalization_1/AssignNewValueAssignVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource>sequential/batch_normalization_1/FusedBatchNormV3:batch_mean:0A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ύ
1sequential/batch_normalization_1/AssignNewValue_1AssignVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceBsequential/batch_normalization_1/FusedBatchNormV3:batch_variance:0C^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
sequential/re_lu_5/ReluRelu5sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????Β
"sequential/max_pooling2d_3/MaxPoolMaxPool%sequential/re_lu_5/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
¦
)sequential/conv2d_6/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0θ
sequential/conv2d_6/Conv2DConv2D+sequential/max_pooling2d_3/MaxPool:output:01sequential/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

*sequential/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ί
sequential/conv2d_6/BiasAddBiasAdd#sequential/conv2d_6/Conv2D:output:02sequential/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
sequential/re_lu_6/ReluRelu$sequential/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????}
IdentityIdentity%sequential/re_lu_6/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????σ	
NoOpNoOp.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_1?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_10^sequential/batch_normalization_1/AssignNewValue2^sequential/batch_normalization_1/AssignNewValue_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp+^sequential/conv2d_3/BiasAdd/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp*^sequential/conv2d_4/Conv2D/ReadVariableOp*^sequential/conv2d_5/Conv2D/ReadVariableOp+^sequential/conv2d_6/BiasAdd/ReadVariableOp*^sequential/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12b
/sequential/batch_normalization_1/AssignNewValue/sequential/batch_normalization_1/AssignNewValue2f
1sequential/batch_normalization_1/AssignNewValue_11sequential/batch_normalization_1/AssignNewValue_12
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_1/ReadVariableOp/sequential/batch_normalization_1/ReadVariableOp2f
1sequential/batch_normalization_1/ReadVariableOp_11sequential/batch_normalization_1/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2X
*sequential/conv2d_3/BiasAdd/ReadVariableOp*sequential/conv2d_3/BiasAdd/ReadVariableOp2V
)sequential/conv2d_3/Conv2D/ReadVariableOp)sequential/conv2d_3/Conv2D/ReadVariableOp2V
)sequential/conv2d_4/Conv2D/ReadVariableOp)sequential/conv2d_4/Conv2D/ReadVariableOp2V
)sequential/conv2d_5/Conv2D/ReadVariableOp)sequential/conv2d_5/Conv2D/ReadVariableOp2X
*sequential/conv2d_6/BiasAdd/ReadVariableOp*sequential/conv2d_6/BiasAdd/ReadVariableOp2V
)sequential/conv2d_6/Conv2D/ReadVariableOp)sequential/conv2d_6/Conv2D/ReadVariableOp:R N
/
_output_shapes
:????????? d

_user_specified_nameX
μ!


C__inference_model_layer_call_and_return_conditional_losses_12613066
x
text9
vgg__feature_extractor_12612948:@-
vgg__feature_extractor_12612950:@:
vgg__feature_extractor_12612952:@.
vgg__feature_extractor_12612954:	;
vgg__feature_extractor_12612956:.
vgg__feature_extractor_12612958:	;
vgg__feature_extractor_12612960:.
vgg__feature_extractor_12612962:	;
vgg__feature_extractor_12612964:.
vgg__feature_extractor_12612966:	.
vgg__feature_extractor_12612968:	.
vgg__feature_extractor_12612970:	.
vgg__feature_extractor_12612972:	;
vgg__feature_extractor_12612974:.
vgg__feature_extractor_12612976:	.
vgg__feature_extractor_12612978:	.
vgg__feature_extractor_12612980:	.
vgg__feature_extractor_12612982:	;
vgg__feature_extractor_12612984:.
vgg__feature_extractor_12612986:	!
dense_12613060:	
dense_12613062:
identity’dense/StatefulPartitionedCall’.vgg__feature_extractor/StatefulPartitionedCall«
.vgg__feature_extractor/StatefulPartitionedCallStatefulPartitionedCallxvgg__feature_extractor_12612948vgg__feature_extractor_12612950vgg__feature_extractor_12612952vgg__feature_extractor_12612954vgg__feature_extractor_12612956vgg__feature_extractor_12612958vgg__feature_extractor_12612960vgg__feature_extractor_12612962vgg__feature_extractor_12612964vgg__feature_extractor_12612966vgg__feature_extractor_12612968vgg__feature_extractor_12612970vgg__feature_extractor_12612972vgg__feature_extractor_12612974vgg__feature_extractor_12612976vgg__feature_extractor_12612978vgg__feature_extractor_12612980vgg__feature_extractor_12612982vgg__feature_extractor_12612984vgg__feature_extractor_12612986* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612626g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             £
	transpose	Transpose7vgg__feature_extractor/StatefulPartitionedCall:output:0transpose/perm:output:0*
T0*0
_output_shapes
:?????????ρ
*adaptive_average_pooling2d/PartitionedCallPartitionedCalltranspose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_12613026
SqueezeSqueeze3adaptive_average_pooling2d/PartitionedCall:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims
ϋ
dense/StatefulPartitionedCallStatefulPartitionedCallSqueeze:output:0dense_12613060dense_12613062*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_12613059y
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????
NoOpNoOp^dense/StatefulPartitionedCall/^vgg__feature_extractor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:????????? d:?????????: : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.vgg__feature_extractor/StatefulPartitionedCall.vgg__feature_extractor/StatefulPartitionedCall:R N
/
_output_shapes
:????????? d

_user_specified_nameX:MI
'
_output_shapes
:?????????

_user_specified_nametext

i
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_12614569

inputs
identity’
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
¨
Ή
F__inference_conv2d_4_layer_call_and_return_conditional_losses_12614401

inputs:
conv2d_readvariableop_resource:
identity’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
μ
Ζ
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12614549

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ϋ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ί
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

Ί
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612761
x-
sequential_12612719:@!
sequential_12612721:@.
sequential_12612723:@"
sequential_12612725:	/
sequential_12612727:"
sequential_12612729:	/
sequential_12612731:"
sequential_12612733:	/
sequential_12612735:"
sequential_12612737:	"
sequential_12612739:	"
sequential_12612741:	"
sequential_12612743:	/
sequential_12612745:"
sequential_12612747:	"
sequential_12612749:	"
sequential_12612751:	"
sequential_12612753:	/
sequential_12612755:"
sequential_12612757:	
identity’"sequential/StatefulPartitionedCall
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_12612719sequential_12612721sequential_12612723sequential_12612725sequential_12612727sequential_12612729sequential_12612731sequential_12612733sequential_12612735sequential_12612737sequential_12612739sequential_12612741sequential_12612743sequential_12612745sequential_12612747sequential_12612749sequential_12612751sequential_12612753sequential_12612755sequential_12612757* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_12612361
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????k
NoOpNoOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:R N
/
_output_shapes
:????????? d

_user_specified_nameX
μ
Y
=__inference_adaptive_average_pooling2d_layer_call_fn_12613919

inputs
identityΟ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_12613026i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
η
_
C__inference_re_lu_layer_call_and_return_conditional_losses_12611946

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? d@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? d@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? d@:W S
/
_output_shapes
:????????? d@
 
_user_specified_nameinputs
Ή
t
X__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_12613954

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0inputs*
T0*Ά
_output_shapes£
 :?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_splitο
stackPacksplit:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7split:output:8split:output:9split:output:10split:output:11split:output:12split:output:13split:output:14split:output:15split:output:16split:output:17split:output:18split:output:19split:output:20split:output:21split:output:22split:output:23*
N*
T0*4
_output_shapes"
 :?????????*

axisS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0stack:output:0*
T0*4
_output_shapes"
 :?????????*
	num_splity
stack_1Packsplit_1:output:0*
N*
T0*8
_output_shapes&
$:"?????????*

axisg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      z
MeanMeanstack_1:output:0Mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ηs
Ϋ
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12613835
xJ
0sequential_conv2d_conv2d_readvariableop_resource:@?
1sequential_conv2d_biasadd_readvariableop_resource:@M
2sequential_conv2d_1_conv2d_readvariableop_resource:@B
3sequential_conv2d_1_biasadd_readvariableop_resource:	N
2sequential_conv2d_2_conv2d_readvariableop_resource:B
3sequential_conv2d_2_biasadd_readvariableop_resource:	N
2sequential_conv2d_3_conv2d_readvariableop_resource:B
3sequential_conv2d_3_biasadd_readvariableop_resource:	N
2sequential_conv2d_4_conv2d_readvariableop_resource:E
6sequential_batch_normalization_readvariableop_resource:	G
8sequential_batch_normalization_readvariableop_1_resource:	V
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:	X
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	N
2sequential_conv2d_5_conv2d_readvariableop_resource:G
8sequential_batch_normalization_1_readvariableop_resource:	I
:sequential_batch_normalization_1_readvariableop_1_resource:	X
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	Z
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	N
2sequential_conv2d_6_conv2d_readvariableop_resource:B
3sequential_conv2d_6_biasadd_readvariableop_resource:	
identity’>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp’@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1’-sequential/batch_normalization/ReadVariableOp’/sequential/batch_normalization/ReadVariableOp_1’@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp’Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1’/sequential/batch_normalization_1/ReadVariableOp’1sequential/batch_normalization_1/ReadVariableOp_1’(sequential/conv2d/BiasAdd/ReadVariableOp’'sequential/conv2d/Conv2D/ReadVariableOp’*sequential/conv2d_1/BiasAdd/ReadVariableOp’)sequential/conv2d_1/Conv2D/ReadVariableOp’*sequential/conv2d_2/BiasAdd/ReadVariableOp’)sequential/conv2d_2/Conv2D/ReadVariableOp’*sequential/conv2d_3/BiasAdd/ReadVariableOp’)sequential/conv2d_3/Conv2D/ReadVariableOp’)sequential/conv2d_4/Conv2D/ReadVariableOp’)sequential/conv2d_5/Conv2D/ReadVariableOp’*sequential/conv2d_6/BiasAdd/ReadVariableOp’)sequential/conv2d_6/Conv2D/ReadVariableOp 
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Έ
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@*
paddingSAME*
strides

(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@{
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? d@½
 sequential/max_pooling2d/MaxPoolMaxPool#sequential/re_lu/Relu:activations:0*/
_output_shapes
:?????????2@*
ksize
*
paddingVALID*
strides
₯
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ε
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2*
paddingSAME*
strides

*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ί
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2Β
"sequential/max_pooling2d_1/MaxPoolMaxPool%sequential/re_lu_1/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
¦
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0η
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ί
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
sequential/re_lu_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????¦
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0α
sequential/conv2d_3/Conv2DConv2D%sequential/re_lu_2/Relu:activations:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

*sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ί
sequential/conv2d_3/BiasAddBiasAdd#sequential/conv2d_3/Conv2D:output:02sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
sequential/re_lu_3/ReluRelu$sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Β
"sequential/max_pooling2d_2/MaxPoolMaxPool%sequential/re_lu_3/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
¦
)sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0η
sequential/conv2d_4/Conv2DConv2D+sequential/max_pooling2d_2/MaxPool:output:01sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
‘
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes	
:*
dtype0₯
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes	
:*
dtype0Γ
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Η
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0σ
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3#sequential/conv2d_4/Conv2D:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 
sequential/re_lu_4/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????¦
)sequential/conv2d_5/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0α
sequential/conv2d_5/Conv2DConv2D%sequential/re_lu_4/Relu:activations:01sequential/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
₯
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0©
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0Η
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Λ
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ύ
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#sequential/conv2d_5/Conv2D:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 
sequential/re_lu_5/ReluRelu5sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????Β
"sequential/max_pooling2d_3/MaxPoolMaxPool%sequential/re_lu_5/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
¦
)sequential/conv2d_6/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0θ
sequential/conv2d_6/Conv2DConv2D+sequential/max_pooling2d_3/MaxPool:output:01sequential/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

*sequential/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ί
sequential/conv2d_6/BiasAddBiasAdd#sequential/conv2d_6/Conv2D:output:02sequential/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
sequential/re_lu_6/ReluRelu$sequential/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????}
IdentityIdentity%sequential/re_lu_6/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????«
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp+^sequential/conv2d_3/BiasAdd/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp*^sequential/conv2d_4/Conv2D/ReadVariableOp*^sequential/conv2d_5/Conv2D/ReadVariableOp+^sequential/conv2d_6/BiasAdd/ReadVariableOp*^sequential/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 2
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_1/ReadVariableOp/sequential/batch_normalization_1/ReadVariableOp2f
1sequential/batch_normalization_1/ReadVariableOp_11sequential/batch_normalization_1/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2X
*sequential/conv2d_3/BiasAdd/ReadVariableOp*sequential/conv2d_3/BiasAdd/ReadVariableOp2V
)sequential/conv2d_3/Conv2D/ReadVariableOp)sequential/conv2d_3/Conv2D/ReadVariableOp2V
)sequential/conv2d_4/Conv2D/ReadVariableOp)sequential/conv2d_4/Conv2D/ReadVariableOp2V
)sequential/conv2d_5/Conv2D/ReadVariableOp)sequential/conv2d_5/Conv2D/ReadVariableOp2X
*sequential/conv2d_6/BiasAdd/ReadVariableOp*sequential/conv2d_6/BiasAdd/ReadVariableOp2V
)sequential/conv2d_6/Conv2D/ReadVariableOp)sequential/conv2d_6/Conv2D/ReadVariableOp:R N
/
_output_shapes
:????????? d

_user_specified_nameX
Ζ
F
*__inference_re_lu_3_layer_call_fn_12614372

inputs
identityΌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_12612017i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
θ!


C__inference_model_layer_call_and_return_conditional_losses_12613187
x
text9
vgg__feature_extractor_12613136:@-
vgg__feature_extractor_12613138:@:
vgg__feature_extractor_12613140:@.
vgg__feature_extractor_12613142:	;
vgg__feature_extractor_12613144:.
vgg__feature_extractor_12613146:	;
vgg__feature_extractor_12613148:.
vgg__feature_extractor_12613150:	;
vgg__feature_extractor_12613152:.
vgg__feature_extractor_12613154:	.
vgg__feature_extractor_12613156:	.
vgg__feature_extractor_12613158:	.
vgg__feature_extractor_12613160:	;
vgg__feature_extractor_12613162:.
vgg__feature_extractor_12613164:	.
vgg__feature_extractor_12613166:	.
vgg__feature_extractor_12613168:	.
vgg__feature_extractor_12613170:	;
vgg__feature_extractor_12613172:.
vgg__feature_extractor_12613174:	!
dense_12613181:	
dense_12613183:
identity’dense/StatefulPartitionedCall’.vgg__feature_extractor/StatefulPartitionedCall§
.vgg__feature_extractor/StatefulPartitionedCallStatefulPartitionedCallxvgg__feature_extractor_12613136vgg__feature_extractor_12613138vgg__feature_extractor_12613140vgg__feature_extractor_12613142vgg__feature_extractor_12613144vgg__feature_extractor_12613146vgg__feature_extractor_12613148vgg__feature_extractor_12613150vgg__feature_extractor_12613152vgg__feature_extractor_12613154vgg__feature_extractor_12613156vgg__feature_extractor_12613158vgg__feature_extractor_12613160vgg__feature_extractor_12613162vgg__feature_extractor_12613164vgg__feature_extractor_12613166vgg__feature_extractor_12613168vgg__feature_extractor_12613170vgg__feature_extractor_12613172vgg__feature_extractor_12613174* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612761g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             £
	transpose	Transpose7vgg__feature_extractor/StatefulPartitionedCall:output:0transpose/perm:output:0*
T0*0
_output_shapes
:?????????ρ
*adaptive_average_pooling2d/PartitionedCallPartitionedCalltranspose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_12613026
SqueezeSqueeze3adaptive_average_pooling2d/PartitionedCall:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims
ϋ
dense/StatefulPartitionedCallStatefulPartitionedCallSqueeze:output:0dense_12613181dense_12613183*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_12613059y
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????
NoOpNoOp^dense/StatefulPartitionedCall/^vgg__feature_extractor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:????????? d:?????????: : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.vgg__feature_extractor/StatefulPartitionedCall.vgg__feature_extractor/StatefulPartitionedCall:R N
/
_output_shapes
:????????? d

_user_specified_nameX:MI
'
_output_shapes
:?????????

_user_specified_nametext
½
Σ
9__inference_vgg__feature_extractor_layer_call_fn_12613756
x!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	&

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	
identity’StatefulPartitionedCallί
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612761x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:????????? d

_user_specified_nameX
Ώ
N
2__inference_max_pooling2d_3_layer_call_fn_12614564

inputs
identityή
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_12611915
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_12614280

inputs
identity’
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
΅


F__inference_conv2d_6_layer_call_and_return_conditional_losses_12612085

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ξ2
ϊ	
!__inference__traced_save_12614688
file_prefix1
-savev2_model_dense_kernel_read_readvariableop/
+savev2_model_dense_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*·
value­BͺB,Prediction/kernel/.ATTRIBUTES/VARIABLE_VALUEB*Prediction/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ϊ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_model_dense_kernel_read_readvariableop+savev2_model_dense_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesϋ
ψ: :	::@:@:@:::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::

_output_shapes
: 
£
ΐ
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612939
input_1-
sequential_12612897:@!
sequential_12612899:@.
sequential_12612901:@"
sequential_12612903:	/
sequential_12612905:"
sequential_12612907:	/
sequential_12612909:"
sequential_12612911:	/
sequential_12612913:"
sequential_12612915:	"
sequential_12612917:	"
sequential_12612919:	"
sequential_12612921:	/
sequential_12612923:"
sequential_12612925:	"
sequential_12612927:	"
sequential_12612929:	"
sequential_12612931:	/
sequential_12612933:"
sequential_12612935:	
identity’"sequential/StatefulPartitionedCall₯
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_12612897sequential_12612899sequential_12612901sequential_12612903sequential_12612905sequential_12612907sequential_12612909sequential_12612911sequential_12612913sequential_12612915sequential_12612917sequential_12612919sequential_12612921sequential_12612923sequential_12612925sequential_12612927sequential_12612929sequential_12612931sequential_12612933sequential_12612935* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_12612361
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????k
NoOpNoOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:X T
/
_output_shapes
:????????? d
!
_user_specified_name	input_1
	
Υ
6__inference_batch_normalization_layer_call_fn_12614427

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12611831
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ώ
N
2__inference_max_pooling2d_2_layer_call_fn_12614382

inputs
identityή
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12611775
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ζS


H__inference_sequential_layer_call_and_return_conditional_losses_12612577
conv2d_input)
conv2d_12612516:@
conv2d_12612518:@,
conv2d_1_12612523:@ 
conv2d_1_12612525:	-
conv2d_2_12612530: 
conv2d_2_12612532:	-
conv2d_3_12612536: 
conv2d_3_12612538:	-
conv2d_4_12612543:+
batch_normalization_12612546:	+
batch_normalization_12612548:	+
batch_normalization_12612550:	+
batch_normalization_12612552:	-
conv2d_5_12612556:-
batch_normalization_1_12612559:	-
batch_normalization_1_12612561:	-
batch_normalization_1_12612563:	-
batch_normalization_1_12612565:	-
conv2d_6_12612570: 
conv2d_6_12612572:	
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’ conv2d_5/StatefulPartitionedCall’ conv2d_6/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_12612516conv2d_12612518*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? d@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_12611935ΰ
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? d@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_12611946η
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????2@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_12611751’
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_12612523conv2d_1_12612525*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_12611959η
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_12611970ξ
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12611763€
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_12612530conv2d_2_12612532*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_12611983η
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_12611994
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_12612536conv2d_3_12612538*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_12612006η
re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_12612017ξ
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12611775
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_4_12612543*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_12612027
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_12612546batch_normalization_12612548batch_normalization_12612550batch_normalization_12612552*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12611831ς
re_lu_4/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_12612045
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_5_12612556*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_12612054
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_1_12612559batch_normalization_1_12612561batch_normalization_1_12612563batch_normalization_1_12612565*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12611895τ
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_12612072ξ
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_12611915€
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_6_12612570conv2d_6_12612572*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_12612085η
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_6_layer_call_and_return_conditional_losses_12612096x
IdentityIdentity re_lu_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:] Y
/
_output_shapes
:????????? d
&
_user_specified_nameconv2d_input

i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12611763

inputs
identity’
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ί

#__inference__wrapped_model_12611742

args_0

args_1g
Mmodel_vgg__feature_extractor_sequential_conv2d_conv2d_readvariableop_resource:@\
Nmodel_vgg__feature_extractor_sequential_conv2d_biasadd_readvariableop_resource:@j
Omodel_vgg__feature_extractor_sequential_conv2d_1_conv2d_readvariableop_resource:@_
Pmodel_vgg__feature_extractor_sequential_conv2d_1_biasadd_readvariableop_resource:	k
Omodel_vgg__feature_extractor_sequential_conv2d_2_conv2d_readvariableop_resource:_
Pmodel_vgg__feature_extractor_sequential_conv2d_2_biasadd_readvariableop_resource:	k
Omodel_vgg__feature_extractor_sequential_conv2d_3_conv2d_readvariableop_resource:_
Pmodel_vgg__feature_extractor_sequential_conv2d_3_biasadd_readvariableop_resource:	k
Omodel_vgg__feature_extractor_sequential_conv2d_4_conv2d_readvariableop_resource:b
Smodel_vgg__feature_extractor_sequential_batch_normalization_readvariableop_resource:	d
Umodel_vgg__feature_extractor_sequential_batch_normalization_readvariableop_1_resource:	s
dmodel_vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:	u
fmodel_vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	k
Omodel_vgg__feature_extractor_sequential_conv2d_5_conv2d_readvariableop_resource:d
Umodel_vgg__feature_extractor_sequential_batch_normalization_1_readvariableop_resource:	f
Wmodel_vgg__feature_extractor_sequential_batch_normalization_1_readvariableop_1_resource:	u
fmodel_vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	w
hmodel_vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	k
Omodel_vgg__feature_extractor_sequential_conv2d_6_conv2d_readvariableop_resource:_
Pmodel_vgg__feature_extractor_sequential_conv2d_6_biasadd_readvariableop_resource:	@
-model_dense_tensordot_readvariableop_resource:	9
+model_dense_biasadd_readvariableop_resource:
identity’"model/dense/BiasAdd/ReadVariableOp’$model/dense/Tensordot/ReadVariableOp’[model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp’]model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1’Jmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp’Lmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1’]model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp’_model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1’Lmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp’Nmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1’Emodel/vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp’Dmodel/vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp’Gmodel/vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp’Fmodel/vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp’Gmodel/vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp’Fmodel/vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp’Gmodel/vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp’Fmodel/vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp’Fmodel/vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp’Fmodel/vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp’Gmodel/vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp’Fmodel/vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOpΪ
Dmodel/vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpMmodel_vgg__feature_extractor_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0χ
5model/vgg__feature_extractor/sequential/conv2d/Conv2DConv2Dargs_0Lmodel/vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@*
paddingSAME*
strides
Π
Emodel/vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpNmodel_vgg__feature_extractor_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
6model/vgg__feature_extractor/sequential/conv2d/BiasAddBiasAdd>model/vgg__feature_extractor/sequential/conv2d/Conv2D:output:0Mmodel/vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@΅
2model/vgg__feature_extractor/sequential/re_lu/ReluRelu?model/vgg__feature_extractor/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? d@χ
=model/vgg__feature_extractor/sequential/max_pooling2d/MaxPoolMaxPool@model/vgg__feature_extractor/sequential/re_lu/Relu:activations:0*/
_output_shapes
:?????????2@*
ksize
*
paddingVALID*
strides
ί
Fmodel/vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOpOmodel_vgg__feature_extractor_sequential_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ό
7model/vgg__feature_extractor/sequential/conv2d_1/Conv2DConv2DFmodel/vgg__feature_extractor/sequential/max_pooling2d/MaxPool:output:0Nmodel/vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2*
paddingSAME*
strides
Υ
Gmodel/vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpPmodel_vgg__feature_extractor_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
8model/vgg__feature_extractor/sequential/conv2d_1/BiasAddBiasAdd@model/vgg__feature_extractor/sequential/conv2d_1/Conv2D:output:0Omodel/vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2Ί
4model/vgg__feature_extractor/sequential/re_lu_1/ReluReluAmodel/vgg__feature_extractor/sequential/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2ό
?model/vgg__feature_extractor/sequential/max_pooling2d_1/MaxPoolMaxPoolBmodel/vgg__feature_extractor/sequential/re_lu_1/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
ΰ
Fmodel/vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOpOmodel_vgg__feature_extractor_sequential_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ύ
7model/vgg__feature_extractor/sequential/conv2d_2/Conv2DConv2DHmodel/vgg__feature_extractor/sequential/max_pooling2d_1/MaxPool:output:0Nmodel/vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
Υ
Gmodel/vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpPmodel_vgg__feature_extractor_sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
8model/vgg__feature_extractor/sequential/conv2d_2/BiasAddBiasAdd@model/vgg__feature_extractor/sequential/conv2d_2/Conv2D:output:0Omodel/vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Ί
4model/vgg__feature_extractor/sequential/re_lu_2/ReluReluAmodel/vgg__feature_extractor/sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ΰ
Fmodel/vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOpOmodel_vgg__feature_extractor_sequential_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Έ
7model/vgg__feature_extractor/sequential/conv2d_3/Conv2DConv2DBmodel/vgg__feature_extractor/sequential/re_lu_2/Relu:activations:0Nmodel/vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
Υ
Gmodel/vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpPmodel_vgg__feature_extractor_sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
8model/vgg__feature_extractor/sequential/conv2d_3/BiasAddBiasAdd@model/vgg__feature_extractor/sequential/conv2d_3/Conv2D:output:0Omodel/vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Ί
4model/vgg__feature_extractor/sequential/re_lu_3/ReluReluAmodel/vgg__feature_extractor/sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ό
?model/vgg__feature_extractor/sequential/max_pooling2d_2/MaxPoolMaxPoolBmodel/vgg__feature_extractor/sequential/re_lu_3/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
ΰ
Fmodel/vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOpOmodel_vgg__feature_extractor_sequential_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ύ
7model/vgg__feature_extractor/sequential/conv2d_4/Conv2DConv2DHmodel/vgg__feature_extractor/sequential/max_pooling2d_2/MaxPool:output:0Nmodel/vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
Ϋ
Jmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOpReadVariableOpSmodel_vgg__feature_extractor_sequential_batch_normalization_readvariableop_resource*
_output_shapes	
:*
dtype0ί
Lmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1ReadVariableOpUmodel_vgg__feature_extractor_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes	
:*
dtype0ύ
[model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpdmodel_vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
]model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpfmodel_vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0‘
Lmodel/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3@model/vgg__feature_extractor/sequential/conv2d_4/Conv2D:output:0Rmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp:value:0Tmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1:value:0cmodel/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0emodel/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( Ι
4model/vgg__feature_extractor/sequential/re_lu_4/ReluReluPmodel/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????ΰ
Fmodel/vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpReadVariableOpOmodel_vgg__feature_extractor_sequential_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Έ
7model/vgg__feature_extractor/sequential/conv2d_5/Conv2DConv2DBmodel/vgg__feature_extractor/sequential/re_lu_4/Relu:activations:0Nmodel/vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
ί
Lmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpReadVariableOpUmodel_vgg__feature_extractor_sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0γ
Nmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOpWmodel_vgg__feature_extractor_sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0
]model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpfmodel_vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
_model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOphmodel_vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0«
Nmodel/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3@model/vgg__feature_extractor/sequential/conv2d_5/Conv2D:output:0Tmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp:value:0Vmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1:value:0emodel/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0gmodel/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( Λ
4model/vgg__feature_extractor/sequential/re_lu_5/ReluReluRmodel/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????ό
?model/vgg__feature_extractor/sequential/max_pooling2d_3/MaxPoolMaxPoolBmodel/vgg__feature_extractor/sequential/re_lu_5/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
ΰ
Fmodel/vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOpReadVariableOpOmodel_vgg__feature_extractor_sequential_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ώ
7model/vgg__feature_extractor/sequential/conv2d_6/Conv2DConv2DHmodel/vgg__feature_extractor/sequential/max_pooling2d_3/MaxPool:output:0Nmodel/vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
Υ
Gmodel/vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpPmodel_vgg__feature_extractor_sequential_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
8model/vgg__feature_extractor/sequential/conv2d_6/BiasAddBiasAdd@model/vgg__feature_extractor/sequential/conv2d_6/Conv2D:output:0Omodel/vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Ί
4model/vgg__feature_extractor/sequential/re_lu_6/ReluReluAmodel/vgg__feature_extractor/sequential/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????m
model/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ί
model/transpose	TransposeBmodel/vgg__feature_extractor/sequential/re_lu_6/Relu:activations:0model/transpose/perm:output:0*
T0*0
_output_shapes
:?????????r
0model/adaptive_average_pooling2d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&model/adaptive_average_pooling2d/splitSplit9model/adaptive_average_pooling2d/split/split_dim:output:0model/transpose:y:0*
T0*Ά
_output_shapes£
 :?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split¨

&model/adaptive_average_pooling2d/stackPack/model/adaptive_average_pooling2d/split:output:0/model/adaptive_average_pooling2d/split:output:1/model/adaptive_average_pooling2d/split:output:2/model/adaptive_average_pooling2d/split:output:3/model/adaptive_average_pooling2d/split:output:4/model/adaptive_average_pooling2d/split:output:5/model/adaptive_average_pooling2d/split:output:6/model/adaptive_average_pooling2d/split:output:7/model/adaptive_average_pooling2d/split:output:8/model/adaptive_average_pooling2d/split:output:90model/adaptive_average_pooling2d/split:output:100model/adaptive_average_pooling2d/split:output:110model/adaptive_average_pooling2d/split:output:120model/adaptive_average_pooling2d/split:output:130model/adaptive_average_pooling2d/split:output:140model/adaptive_average_pooling2d/split:output:150model/adaptive_average_pooling2d/split:output:160model/adaptive_average_pooling2d/split:output:170model/adaptive_average_pooling2d/split:output:180model/adaptive_average_pooling2d/split:output:190model/adaptive_average_pooling2d/split:output:200model/adaptive_average_pooling2d/split:output:210model/adaptive_average_pooling2d/split:output:220model/adaptive_average_pooling2d/split:output:23*
N*
T0*4
_output_shapes"
 :?????????*

axist
2model/adaptive_average_pooling2d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ο
(model/adaptive_average_pooling2d/split_1Split;model/adaptive_average_pooling2d/split_1/split_dim:output:0/model/adaptive_average_pooling2d/stack:output:0*
T0*4
_output_shapes"
 :?????????*
	num_split»
(model/adaptive_average_pooling2d/stack_1Pack1model/adaptive_average_pooling2d/split_1:output:0*
N*
T0*8
_output_shapes&
$:"?????????*

axis
7model/adaptive_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      έ
%model/adaptive_average_pooling2d/MeanMean1model/adaptive_average_pooling2d/stack_1:output:0@model/adaptive_average_pooling2d/Mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????
model/SqueezeSqueeze.model/adaptive_average_pooling2d/Mean:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

$model/dense/Tensordot/ReadVariableOpReadVariableOp-model_dense_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0d
model/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
model/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       a
model/dense/Tensordot/ShapeShapemodel/Squeeze:output:0*
T0*
_output_shapes
:e
#model/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : λ
model/dense/Tensordot/GatherV2GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/free:output:0,model/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
%model/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ο
 model/dense/Tensordot/GatherV2_1GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/axes:output:0.model/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
model/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense/Tensordot/ProdProd'model/dense/Tensordot/GatherV2:output:0$model/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: g
model/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense/Tensordot/Prod_1Prod)model/dense/Tensordot/GatherV2_1:output:0&model/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!model/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Μ
model/dense/Tensordot/concatConcatV2#model/dense/Tensordot/free:output:0#model/dense/Tensordot/axes:output:0*model/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
model/dense/Tensordot/stackPack#model/dense/Tensordot/Prod:output:0%model/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:’
model/dense/Tensordot/transpose	Transposemodel/Squeeze:output:0%model/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????
model/dense/Tensordot/ReshapeReshape#model/dense/Tensordot/transpose:y:0$model/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
model/dense/Tensordot/MatMulMatMul&model/dense/Tensordot/Reshape:output:0,model/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
model/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:e
#model/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Χ
model/dense/Tensordot/concat_1ConcatV2'model/dense/Tensordot/GatherV2:output:0&model/dense/Tensordot/Const_2:output:0,model/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:§
model/dense/TensordotReshape&model/dense/Tensordot/MatMul:product:0'model/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense/BiasAddBiasAddmodel/dense/Tensordot:output:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????o
IdentityIdentitymodel/dense/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????»
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/Tensordot/ReadVariableOp\^model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp^^model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1K^model/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOpM^model/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1^^model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp`^model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1M^model/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpO^model/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1F^model/vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOpE^model/vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpH^model/vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpG^model/vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpH^model/vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpG^model/vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpH^model/vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpG^model/vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpG^model/vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpG^model/vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpH^model/vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpG^model/vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:????????? d:?????????: : : : : : : : : : : : : : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/Tensordot/ReadVariableOp$model/dense/Tensordot/ReadVariableOp2Ί
[model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp[model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2Ύ
]model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1]model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12
Jmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOpJmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp2
Lmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1Lmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_12Ύ
]model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp]model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2Β
_model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1_model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12
Lmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpLmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp2 
Nmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1Nmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_12
Emodel/vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOpEmodel/vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp2
Dmodel/vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpDmodel/vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp2
Gmodel/vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpGmodel/vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp2
Fmodel/vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpFmodel/vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp2
Gmodel/vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpGmodel/vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp2
Fmodel/vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpFmodel/vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp2
Gmodel/vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpGmodel/vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp2
Fmodel/vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpFmodel/vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp2
Fmodel/vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpFmodel/vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp2
Fmodel/vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpFmodel/vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp2
Gmodel/vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpGmodel/vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp2
Fmodel/vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOpFmodel/vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? d
 
_user_specified_nameargs_0:OK
'
_output_shapes
:?????????
 
_user_specified_nameargs_1
Κ
?
-__inference_sequential_layer_call_fn_12612142
conv2d_input!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	&

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	
identity’StatefulPartitionedCallβ
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_12612099x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:????????? d
&
_user_specified_nameconv2d_input

Ί
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612626
x-
sequential_12612584:@!
sequential_12612586:@.
sequential_12612588:@"
sequential_12612590:	/
sequential_12612592:"
sequential_12612594:	/
sequential_12612596:"
sequential_12612598:	/
sequential_12612600:"
sequential_12612602:	"
sequential_12612604:	"
sequential_12612606:	"
sequential_12612608:	/
sequential_12612610:"
sequential_12612612:	"
sequential_12612614:	"
sequential_12612616:	"
sequential_12612618:	/
sequential_12612620:"
sequential_12612622:	
identity’"sequential/StatefulPartitionedCall£
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_12612584sequential_12612586sequential_12612588sequential_12612590sequential_12612592sequential_12612594sequential_12612596sequential_12612598sequential_12612600sequential_12612602sequential_12612604sequential_12612606sequential_12612608sequential_12612610sequential_12612612sequential_12612614sequential_12612616sequential_12612618sequential_12612620sequential_12612622* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_12612099
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????k
NoOpNoOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:R N
/
_output_shapes
:????????? d

_user_specified_nameX
°


F__inference_conv2d_1_layer_call_and_return_conditional_losses_12614299

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????2w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????2@
 
_user_specified_nameinputs
ψ
£
+__inference_conv2d_6_layer_call_fn_12614578

inputs#
unknown:
	unknown_0:	
identity’StatefulPartitionedCallη
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_12612085x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ν
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_12614348

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Φ

(__inference_dense_layer_call_fn_12613963

inputs
unknown:	
	unknown_0:
identity’StatefulPartitionedCallί
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_12613059s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
§ζ
Α
C__inference_model_layer_call_and_return_conditional_losses_12613614
x
texta
Gvgg__feature_extractor_sequential_conv2d_conv2d_readvariableop_resource:@V
Hvgg__feature_extractor_sequential_conv2d_biasadd_readvariableop_resource:@d
Ivgg__feature_extractor_sequential_conv2d_1_conv2d_readvariableop_resource:@Y
Jvgg__feature_extractor_sequential_conv2d_1_biasadd_readvariableop_resource:	e
Ivgg__feature_extractor_sequential_conv2d_2_conv2d_readvariableop_resource:Y
Jvgg__feature_extractor_sequential_conv2d_2_biasadd_readvariableop_resource:	e
Ivgg__feature_extractor_sequential_conv2d_3_conv2d_readvariableop_resource:Y
Jvgg__feature_extractor_sequential_conv2d_3_biasadd_readvariableop_resource:	e
Ivgg__feature_extractor_sequential_conv2d_4_conv2d_readvariableop_resource:\
Mvgg__feature_extractor_sequential_batch_normalization_readvariableop_resource:	^
Ovgg__feature_extractor_sequential_batch_normalization_readvariableop_1_resource:	m
^vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:	o
`vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	e
Ivgg__feature_extractor_sequential_conv2d_5_conv2d_readvariableop_resource:^
Ovgg__feature_extractor_sequential_batch_normalization_1_readvariableop_resource:	`
Qvgg__feature_extractor_sequential_batch_normalization_1_readvariableop_1_resource:	o
`vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	q
bvgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	e
Ivgg__feature_extractor_sequential_conv2d_6_conv2d_readvariableop_resource:Y
Jvgg__feature_extractor_sequential_conv2d_6_biasadd_readvariableop_resource:	:
'dense_tensordot_readvariableop_resource:	3
%dense_biasadd_readvariableop_resource:
identity’dense/BiasAdd/ReadVariableOp’dense/Tensordot/ReadVariableOp’Dvgg__feature_extractor/sequential/batch_normalization/AssignNewValue’Fvgg__feature_extractor/sequential/batch_normalization/AssignNewValue_1’Uvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp’Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1’Dvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp’Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1’Fvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue’Hvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue_1’Wvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp’Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1’Fvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp’Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1’?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp’>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp’Avgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp’@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp’Avgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp’@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp’Avgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp’@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp’@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp’@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp’Avgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp’@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOpΞ
>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpGvgg__feature_extractor_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ζ
/vgg__feature_extractor/sequential/conv2d/Conv2DConv2DxFvgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@*
paddingSAME*
strides
Δ
?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpHvgg__feature_extractor_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ψ
0vgg__feature_extractor/sequential/conv2d/BiasAddBiasAdd8vgg__feature_extractor/sequential/conv2d/Conv2D:output:0Gvgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@©
,vgg__feature_extractor/sequential/re_lu/ReluRelu9vgg__feature_extractor/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? d@λ
7vgg__feature_extractor/sequential/max_pooling2d/MaxPoolMaxPool:vgg__feature_extractor/sequential/re_lu/Relu:activations:0*/
_output_shapes
:?????????2@*
ksize
*
paddingVALID*
strides
Σ
@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ͺ
1vgg__feature_extractor/sequential/conv2d_1/Conv2DConv2D@vgg__feature_extractor/sequential/max_pooling2d/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2*
paddingSAME*
strides
Ι
Avgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0?
2vgg__feature_extractor/sequential/conv2d_1/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_1/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?
.vgg__feature_extractor/sequential/re_lu_1/ReluRelu;vgg__feature_extractor/sequential/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2π
9vgg__feature_extractor/sequential/max_pooling2d_1/MaxPoolMaxPool<vgg__feature_extractor/sequential/re_lu_1/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
Τ
@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
1vgg__feature_extractor/sequential/conv2d_2/Conv2DConv2DBvgg__feature_extractor/sequential/max_pooling2d_1/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
Ι
Avgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0?
2vgg__feature_extractor/sequential/conv2d_2/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_2/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
.vgg__feature_extractor/sequential/re_lu_2/ReluRelu;vgg__feature_extractor/sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Τ
@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¦
1vgg__feature_extractor/sequential/conv2d_3/Conv2DConv2D<vgg__feature_extractor/sequential/re_lu_2/Relu:activations:0Hvgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
Ι
Avgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0?
2vgg__feature_extractor/sequential/conv2d_3/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_3/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
.vgg__feature_extractor/sequential/re_lu_3/ReluRelu;vgg__feature_extractor/sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????π
9vgg__feature_extractor/sequential/max_pooling2d_2/MaxPoolMaxPool<vgg__feature_extractor/sequential/re_lu_3/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
Τ
@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
1vgg__feature_extractor/sequential/conv2d_4/Conv2DConv2DBvgg__feature_extractor/sequential/max_pooling2d_2/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
Ο
Dvgg__feature_extractor/sequential/batch_normalization/ReadVariableOpReadVariableOpMvgg__feature_extractor_sequential_batch_normalization_readvariableop_resource*
_output_shapes	
:*
dtype0Σ
Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1ReadVariableOpOvgg__feature_extractor_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes	
:*
dtype0ρ
Uvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp^vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0υ
Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
Fvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3:vgg__feature_extractor/sequential/conv2d_4/Conv2D:output:0Lvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp:value:0Nvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1:value:0]vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0_vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<
Dvgg__feature_extractor/sequential/batch_normalization/AssignNewValueAssignVariableOp^vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceSvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3:batch_mean:0V^vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
Fvgg__feature_extractor/sequential/batch_normalization/AssignNewValue_1AssignVariableOp`vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceWvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3:batch_variance:0X^vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0½
.vgg__feature_extractor/sequential/re_lu_4/ReluReluJvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????Τ
@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¦
1vgg__feature_extractor/sequential/conv2d_5/Conv2DConv2D<vgg__feature_extractor/sequential/re_lu_4/Relu:activations:0Hvgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
Σ
Fvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpReadVariableOpOvgg__feature_extractor_sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0Χ
Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOpQvgg__feature_extractor_sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0υ
Wvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp`vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ω
Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbvgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
Hvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3:vgg__feature_extractor/sequential/conv2d_5/Conv2D:output:0Nvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp:value:0Pvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1:value:0_vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0avgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<
Fvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValueAssignVariableOp`vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceUvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3:batch_mean:0X^vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
Hvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue_1AssignVariableOpbvgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceYvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3:batch_variance:0Z^vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0Ώ
.vgg__feature_extractor/sequential/re_lu_5/ReluReluLvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????π
9vgg__feature_extractor/sequential/max_pooling2d_3/MaxPoolMaxPool<vgg__feature_extractor/sequential/re_lu_5/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
Τ
@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0­
1vgg__feature_extractor/sequential/conv2d_6/Conv2DConv2DBvgg__feature_extractor/sequential/max_pooling2d_3/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
Ι
Avgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0?
2vgg__feature_extractor/sequential/conv2d_6/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_6/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
.vgg__feature_extractor/sequential/re_lu_6/ReluRelu;vgg__feature_extractor/sequential/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¨
	transpose	Transpose<vgg__feature_extractor/sequential/re_lu_6/Relu:activations:0transpose/perm:output:0*
T0*0
_output_shapes
:?????????l
*adaptive_average_pooling2d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ΐ
 adaptive_average_pooling2d/splitSplit3adaptive_average_pooling2d/split/split_dim:output:0transpose:y:0*
T0*Ά
_output_shapes£
 :?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split	
 adaptive_average_pooling2d/stackPack)adaptive_average_pooling2d/split:output:0)adaptive_average_pooling2d/split:output:1)adaptive_average_pooling2d/split:output:2)adaptive_average_pooling2d/split:output:3)adaptive_average_pooling2d/split:output:4)adaptive_average_pooling2d/split:output:5)adaptive_average_pooling2d/split:output:6)adaptive_average_pooling2d/split:output:7)adaptive_average_pooling2d/split:output:8)adaptive_average_pooling2d/split:output:9*adaptive_average_pooling2d/split:output:10*adaptive_average_pooling2d/split:output:11*adaptive_average_pooling2d/split:output:12*adaptive_average_pooling2d/split:output:13*adaptive_average_pooling2d/split:output:14*adaptive_average_pooling2d/split:output:15*adaptive_average_pooling2d/split:output:16*adaptive_average_pooling2d/split:output:17*adaptive_average_pooling2d/split:output:18*adaptive_average_pooling2d/split:output:19*adaptive_average_pooling2d/split:output:20*adaptive_average_pooling2d/split:output:21*adaptive_average_pooling2d/split:output:22*adaptive_average_pooling2d/split:output:23*
N*
T0*4
_output_shapes"
 :?????????*

axisn
,adaptive_average_pooling2d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :έ
"adaptive_average_pooling2d/split_1Split5adaptive_average_pooling2d/split_1/split_dim:output:0)adaptive_average_pooling2d/stack:output:0*
T0*4
_output_shapes"
 :?????????*
	num_split―
"adaptive_average_pooling2d/stack_1Pack+adaptive_average_pooling2d/split_1:output:0*
N*
T0*8
_output_shapes&
$:"?????????*

axis
1adaptive_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Λ
adaptive_average_pooling2d/MeanMean+adaptive_average_pooling2d/stack_1:output:0:adaptive_average_pooling2d/Mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????
SqueezeSqueeze(adaptive_average_pooling2d/Mean:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       U
dense/Tensordot/ShapeShapeSqueeze:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Σ
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Χ
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ΄
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	TransposeSqueeze:output:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ώ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????i
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????Ϋ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOpE^vgg__feature_extractor/sequential/batch_normalization/AssignNewValueG^vgg__feature_extractor/sequential/batch_normalization/AssignNewValue_1V^vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpX^vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1E^vgg__feature_extractor/sequential/batch_normalization/ReadVariableOpG^vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1G^vgg__feature_extractor/sequential/batch_normalization_1/AssignNewValueI^vgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue_1X^vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpZ^vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1G^vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpI^vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1@^vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp?^vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:????????? d:?????????: : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2
Dvgg__feature_extractor/sequential/batch_normalization/AssignNewValueDvgg__feature_extractor/sequential/batch_normalization/AssignNewValue2
Fvgg__feature_extractor/sequential/batch_normalization/AssignNewValue_1Fvgg__feature_extractor/sequential/batch_normalization/AssignNewValue_12?
Uvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpUvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2²
Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12
Dvgg__feature_extractor/sequential/batch_normalization/ReadVariableOpDvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp2
Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_12
Fvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValueFvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue2
Hvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue_1Hvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue_12²
Wvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpWvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2Ά
Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12
Fvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpFvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp2
Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_12
?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp2
>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp2
Avgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp2
@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp2
Avgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp2
@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp2
Avgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp2
@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp2
@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp2
@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp2
Avgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp2
@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp:R N
/
_output_shapes
:????????? d

_user_specified_nameX:MI
'
_output_shapes
:?????????

_user_specified_nametext
΄


F__inference_conv2d_2_layer_call_and_return_conditional_losses_12614338

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ώ
N
2__inference_max_pooling2d_1_layer_call_fn_12614314

inputs
identityή
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12611763
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ν
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_12611994

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ή
’
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12611864

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ά
 
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12614445

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
κ
Δ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12611831

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ϋ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ί
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ζ
F
*__inference_re_lu_1_layer_call_fn_12614304

inputs
identityΌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_12611970i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2:X T
0
_output_shapes
:?????????2
 
_user_specified_nameinputs
κ
Δ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12614463

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ϋ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ί
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
΄


F__inference_conv2d_3_layer_call_and_return_conditional_losses_12612006

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
§

ύ
D__inference_conv2d_layer_call_and_return_conditional_losses_12611935

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? d@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? d
 
_user_specified_nameinputs
΅


F__inference_conv2d_6_layer_call_and_return_conditional_losses_12614588

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ν
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_12611970

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????2c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2:X T
0
_output_shapes
:?????????2
 
_user_specified_nameinputs
ν
a
E__inference_re_lu_6_layer_call_and_return_conditional_losses_12612096

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ν
a
E__inference_re_lu_4_layer_call_and_return_conditional_losses_12614473

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
΄
Μ
-__inference_sequential_layer_call_fn_12614083

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	&

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	
identity’StatefulPartitionedCallΨ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_12612361x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? d
 
_user_specified_nameinputs
η
_
C__inference_re_lu_layer_call_and_return_conditional_losses_12614270

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? d@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? d@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? d@:W S
/
_output_shapes
:????????? d@
 
_user_specified_nameinputs
ν
a
E__inference_re_lu_3_layer_call_and_return_conditional_losses_12614377

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ίa

H__inference_sequential_layer_call_and_return_conditional_losses_12614162

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@B
'conv2d_1_conv2d_readvariableop_resource:@7
(conv2d_1_biasadd_readvariableop_resource:	C
'conv2d_2_conv2d_readvariableop_resource:7
(conv2d_2_biasadd_readvariableop_resource:	C
'conv2d_3_conv2d_readvariableop_resource:7
(conv2d_3_biasadd_readvariableop_resource:	C
'conv2d_4_conv2d_readvariableop_resource::
+batch_normalization_readvariableop_resource:	<
-batch_normalization_readvariableop_1_resource:	K
<batch_normalization_fusedbatchnormv3_readvariableop_resource:	M
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_5_conv2d_readvariableop_resource:<
-batch_normalization_1_readvariableop_resource:	>
/batch_normalization_1_readvariableop_1_resource:	M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_6_conv2d_readvariableop_resource:7
(conv2d_6_biasadd_readvariableop_resource:	
identity’3batch_normalization/FusedBatchNormV3/ReadVariableOp’5batch_normalization/FusedBatchNormV3/ReadVariableOp_1’"batch_normalization/ReadVariableOp’$batch_normalization/ReadVariableOp_1’5batch_normalization_1/FusedBatchNormV3/ReadVariableOp’7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1’$batch_normalization_1/ReadVariableOp’&batch_normalization_1/ReadVariableOp_1’conv2d/BiasAdd/ReadVariableOp’conv2d/Conv2D/ReadVariableOp’conv2d_1/BiasAdd/ReadVariableOp’conv2d_1/Conv2D/ReadVariableOp’conv2d_2/BiasAdd/ReadVariableOp’conv2d_2/Conv2D/ReadVariableOp’conv2d_3/BiasAdd/ReadVariableOp’conv2d_3/Conv2D/ReadVariableOp’conv2d_4/Conv2D/ReadVariableOp’conv2d_5/Conv2D/ReadVariableOp’conv2d_6/BiasAdd/ReadVariableOp’conv2d_6/Conv2D/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0§
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@*
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@e

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? d@§
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:?????????2@*
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Δ
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2*
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2j
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2¬
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ζ
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????j
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ΐ
conv2d_3/Conv2DConv2Dre_lu_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????j
re_lu_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????¬
max_pooling2d_2/MaxPoolMaxPoolre_lu_3/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides

conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ζ
conv2d_4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:*
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:*
dtype0­
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0±
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( y
re_lu_4/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ΐ
conv2d_5/Conv2DConv2Dre_lu_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0΅
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0»
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( {
re_lu_5/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????¬
max_pooling2d_3/MaxPoolMaxPoolre_lu_5/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides

conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Η
conv2d_6/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????j
re_lu_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????r
IdentityIdentityre_lu_6/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????Ο
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? d
 
_user_specified_nameinputs
ν
a
E__inference_re_lu_6_layer_call_and_return_conditional_losses_12614598

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ν
a
E__inference_re_lu_4_layer_call_and_return_conditional_losses_12612045

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ζ
F
*__inference_re_lu_6_layer_call_fn_12614593

inputs
identityΌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_6_layer_call_and_return_conditional_losses_12612096i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_layer_call_fn_12614275

inputs
identityά
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_12611751
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Ζ
F
*__inference_re_lu_4_layer_call_fn_12614468

inputs
identityΌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_12612045i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ψ
£
+__inference_conv2d_3_layer_call_fn_12614357

inputs#
unknown:
	unknown_0:	
identity’StatefulPartitionedCallη
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_12612006x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ζ
F
*__inference_re_lu_2_layer_call_fn_12614343

inputs
identityΌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_12611994i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
υ
’
+__inference_conv2d_1_layer_call_fn_12614289

inputs"
unknown:@
	unknown_0:	
identity’StatefulPartitionedCallη
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_12611959x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????2@
 
_user_specified_nameinputs
΄


F__inference_conv2d_3_layer_call_and_return_conditional_losses_12614367

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ο
ϋ
C__inference_dense_layer_call_and_return_conditional_losses_12613993

inputs4
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ώ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:?????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12614387

inputs
identity’
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Ζ
F
*__inference_re_lu_5_layer_call_fn_12614554

inputs
identityΌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_12612072i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
εW
Υ
$__inference__traced_restore_12614764
file_prefix6
#assignvariableop_model_dense_kernel:	1
#assignvariableop_1_model_dense_bias::
 assignvariableop_2_conv2d_kernel:@,
assignvariableop_3_conv2d_bias:@=
"assignvariableop_4_conv2d_1_kernel:@/
 assignvariableop_5_conv2d_1_bias:	>
"assignvariableop_6_conv2d_2_kernel:/
 assignvariableop_7_conv2d_2_bias:	>
"assignvariableop_8_conv2d_3_kernel:/
 assignvariableop_9_conv2d_3_bias:	?
#assignvariableop_10_conv2d_4_kernel:<
-assignvariableop_11_batch_normalization_gamma:	;
,assignvariableop_12_batch_normalization_beta:	B
3assignvariableop_13_batch_normalization_moving_mean:	F
7assignvariableop_14_batch_normalization_moving_variance:	?
#assignvariableop_15_conv2d_5_kernel:>
/assignvariableop_16_batch_normalization_1_gamma:	=
.assignvariableop_17_batch_normalization_1_beta:	D
5assignvariableop_18_batch_normalization_1_moving_mean:	H
9assignvariableop_19_batch_normalization_1_moving_variance:	?
#assignvariableop_20_conv2d_6_kernel:0
!assignvariableop_21_conv2d_6_bias:	
identity_23’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*·
value­BͺB,Prediction/kernel/.ATTRIBUTES/VARIABLE_VALUEB*Prediction/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp#assignvariableop_model_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp#assignvariableop_1_model_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp-assignvariableop_11_batch_normalization_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp,assignvariableop_12_batch_normalization_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_13AssignVariableOp3assignvariableop_13_batch_normalization_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_14AssignVariableOp7assignvariableop_14_batch_normalization_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv2d_5_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_1_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_1_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_1_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_1_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv2d_6_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv2d_6_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ³
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
:  
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ή
’
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12614531

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ν
a
E__inference_re_lu_5_layer_call_and_return_conditional_losses_12612072

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ύ
D
(__inference_re_lu_layer_call_fn_12614265

inputs
identityΉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? d@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_12611946h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? d@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? d@:W S
/
_output_shapes
:????????? d@
 
_user_specified_nameinputs
ν
a
E__inference_re_lu_5_layer_call_and_return_conditional_losses_12614559

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ο
ϋ
C__inference_dense_layer_call_and_return_conditional_losses_12613059

inputs4
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ώ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:?????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
 	
Χ
8__inference_batch_normalization_1_layer_call_fn_12614500

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity’StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12611864
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
§Π

C__inference_model_layer_call_and_return_conditional_losses_12613474
x
texta
Gvgg__feature_extractor_sequential_conv2d_conv2d_readvariableop_resource:@V
Hvgg__feature_extractor_sequential_conv2d_biasadd_readvariableop_resource:@d
Ivgg__feature_extractor_sequential_conv2d_1_conv2d_readvariableop_resource:@Y
Jvgg__feature_extractor_sequential_conv2d_1_biasadd_readvariableop_resource:	e
Ivgg__feature_extractor_sequential_conv2d_2_conv2d_readvariableop_resource:Y
Jvgg__feature_extractor_sequential_conv2d_2_biasadd_readvariableop_resource:	e
Ivgg__feature_extractor_sequential_conv2d_3_conv2d_readvariableop_resource:Y
Jvgg__feature_extractor_sequential_conv2d_3_biasadd_readvariableop_resource:	e
Ivgg__feature_extractor_sequential_conv2d_4_conv2d_readvariableop_resource:\
Mvgg__feature_extractor_sequential_batch_normalization_readvariableop_resource:	^
Ovgg__feature_extractor_sequential_batch_normalization_readvariableop_1_resource:	m
^vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:	o
`vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	e
Ivgg__feature_extractor_sequential_conv2d_5_conv2d_readvariableop_resource:^
Ovgg__feature_extractor_sequential_batch_normalization_1_readvariableop_resource:	`
Qvgg__feature_extractor_sequential_batch_normalization_1_readvariableop_1_resource:	o
`vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	q
bvgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	e
Ivgg__feature_extractor_sequential_conv2d_6_conv2d_readvariableop_resource:Y
Jvgg__feature_extractor_sequential_conv2d_6_biasadd_readvariableop_resource:	:
'dense_tensordot_readvariableop_resource:	3
%dense_biasadd_readvariableop_resource:
identity’dense/BiasAdd/ReadVariableOp’dense/Tensordot/ReadVariableOp’Uvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp’Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1’Dvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp’Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1’Wvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp’Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1’Fvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp’Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1’?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp’>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp’Avgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp’@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp’Avgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp’@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp’Avgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp’@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp’@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp’@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp’Avgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp’@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOpΞ
>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpGvgg__feature_extractor_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ζ
/vgg__feature_extractor/sequential/conv2d/Conv2DConv2DxFvgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@*
paddingSAME*
strides
Δ
?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpHvgg__feature_extractor_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ψ
0vgg__feature_extractor/sequential/conv2d/BiasAddBiasAdd8vgg__feature_extractor/sequential/conv2d/Conv2D:output:0Gvgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@©
,vgg__feature_extractor/sequential/re_lu/ReluRelu9vgg__feature_extractor/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? d@λ
7vgg__feature_extractor/sequential/max_pooling2d/MaxPoolMaxPool:vgg__feature_extractor/sequential/re_lu/Relu:activations:0*/
_output_shapes
:?????????2@*
ksize
*
paddingVALID*
strides
Σ
@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ͺ
1vgg__feature_extractor/sequential/conv2d_1/Conv2DConv2D@vgg__feature_extractor/sequential/max_pooling2d/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2*
paddingSAME*
strides
Ι
Avgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0?
2vgg__feature_extractor/sequential/conv2d_1/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_1/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?
.vgg__feature_extractor/sequential/re_lu_1/ReluRelu;vgg__feature_extractor/sequential/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2π
9vgg__feature_extractor/sequential/max_pooling2d_1/MaxPoolMaxPool<vgg__feature_extractor/sequential/re_lu_1/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
Τ
@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
1vgg__feature_extractor/sequential/conv2d_2/Conv2DConv2DBvgg__feature_extractor/sequential/max_pooling2d_1/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
Ι
Avgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0?
2vgg__feature_extractor/sequential/conv2d_2/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_2/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
.vgg__feature_extractor/sequential/re_lu_2/ReluRelu;vgg__feature_extractor/sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????Τ
@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¦
1vgg__feature_extractor/sequential/conv2d_3/Conv2DConv2D<vgg__feature_extractor/sequential/re_lu_2/Relu:activations:0Hvgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
Ι
Avgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0?
2vgg__feature_extractor/sequential/conv2d_3/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_3/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
.vgg__feature_extractor/sequential/re_lu_3/ReluRelu;vgg__feature_extractor/sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????π
9vgg__feature_extractor/sequential/max_pooling2d_2/MaxPoolMaxPool<vgg__feature_extractor/sequential/re_lu_3/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
Τ
@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
1vgg__feature_extractor/sequential/conv2d_4/Conv2DConv2DBvgg__feature_extractor/sequential/max_pooling2d_2/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
Ο
Dvgg__feature_extractor/sequential/batch_normalization/ReadVariableOpReadVariableOpMvgg__feature_extractor_sequential_batch_normalization_readvariableop_resource*
_output_shapes	
:*
dtype0Σ
Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1ReadVariableOpOvgg__feature_extractor_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes	
:*
dtype0ρ
Uvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp^vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0υ
Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ύ
Fvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3:vgg__feature_extractor/sequential/conv2d_4/Conv2D:output:0Lvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp:value:0Nvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1:value:0]vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0_vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( ½
.vgg__feature_extractor/sequential/re_lu_4/ReluReluJvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????Τ
@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¦
1vgg__feature_extractor/sequential/conv2d_5/Conv2DConv2D<vgg__feature_extractor/sequential/re_lu_4/Relu:activations:0Hvgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
Σ
Fvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpReadVariableOpOvgg__feature_extractor_sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0Χ
Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOpQvgg__feature_extractor_sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0υ
Wvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp`vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ω
Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbvgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
Hvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3:vgg__feature_extractor/sequential/conv2d_5/Conv2D:output:0Nvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp:value:0Pvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1:value:0_vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0avgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( Ώ
.vgg__feature_extractor/sequential/re_lu_5/ReluReluLvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????π
9vgg__feature_extractor/sequential/max_pooling2d_3/MaxPoolMaxPool<vgg__feature_extractor/sequential/re_lu_5/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
Τ
@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0­
1vgg__feature_extractor/sequential/conv2d_6/Conv2DConv2DBvgg__feature_extractor/sequential/max_pooling2d_3/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
Ι
Avgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0?
2vgg__feature_extractor/sequential/conv2d_6/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_6/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
.vgg__feature_extractor/sequential/re_lu_6/ReluRelu;vgg__feature_extractor/sequential/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¨
	transpose	Transpose<vgg__feature_extractor/sequential/re_lu_6/Relu:activations:0transpose/perm:output:0*
T0*0
_output_shapes
:?????????l
*adaptive_average_pooling2d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ΐ
 adaptive_average_pooling2d/splitSplit3adaptive_average_pooling2d/split/split_dim:output:0transpose:y:0*
T0*Ά
_output_shapes£
 :?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split	
 adaptive_average_pooling2d/stackPack)adaptive_average_pooling2d/split:output:0)adaptive_average_pooling2d/split:output:1)adaptive_average_pooling2d/split:output:2)adaptive_average_pooling2d/split:output:3)adaptive_average_pooling2d/split:output:4)adaptive_average_pooling2d/split:output:5)adaptive_average_pooling2d/split:output:6)adaptive_average_pooling2d/split:output:7)adaptive_average_pooling2d/split:output:8)adaptive_average_pooling2d/split:output:9*adaptive_average_pooling2d/split:output:10*adaptive_average_pooling2d/split:output:11*adaptive_average_pooling2d/split:output:12*adaptive_average_pooling2d/split:output:13*adaptive_average_pooling2d/split:output:14*adaptive_average_pooling2d/split:output:15*adaptive_average_pooling2d/split:output:16*adaptive_average_pooling2d/split:output:17*adaptive_average_pooling2d/split:output:18*adaptive_average_pooling2d/split:output:19*adaptive_average_pooling2d/split:output:20*adaptive_average_pooling2d/split:output:21*adaptive_average_pooling2d/split:output:22*adaptive_average_pooling2d/split:output:23*
N*
T0*4
_output_shapes"
 :?????????*

axisn
,adaptive_average_pooling2d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :έ
"adaptive_average_pooling2d/split_1Split5adaptive_average_pooling2d/split_1/split_dim:output:0)adaptive_average_pooling2d/stack:output:0*
T0*4
_output_shapes"
 :?????????*
	num_split―
"adaptive_average_pooling2d/stack_1Pack+adaptive_average_pooling2d/split_1:output:0*
N*
T0*8
_output_shapes&
$:"?????????*

axis
1adaptive_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Λ
adaptive_average_pooling2d/MeanMean+adaptive_average_pooling2d/stack_1:output:0:adaptive_average_pooling2d/Mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????
SqueezeSqueeze(adaptive_average_pooling2d/Mean:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       U
dense/Tensordot/ShapeShapeSqueeze:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Σ
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Χ
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ΄
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	TransposeSqueeze:output:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ώ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????i
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????·
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOpV^vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpX^vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1E^vgg__feature_extractor/sequential/batch_normalization/ReadVariableOpG^vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1X^vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpZ^vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1G^vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpI^vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1@^vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp?^vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:????????? d:?????????: : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2?
Uvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpUvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2²
Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12
Dvgg__feature_extractor/sequential/batch_normalization/ReadVariableOpDvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp2
Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_12²
Wvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpWvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2Ά
Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12
Fvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpFvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp2
Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_12
?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp2
>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp2
Avgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp2
@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp2
Avgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp2
@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp2
Avgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp2
@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp2
@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp2
@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp2
Avgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp2
@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp:R N
/
_output_shapes
:????????? d

_user_specified_nameX:MI
'
_output_shapes
:?????????

_user_specified_nametext
	
Υ
6__inference_batch_normalization_layer_call_fn_12614414

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12611800
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
α

(__inference_model_layer_call_fn_12613284
x
text!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	&

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:
identity’StatefulPartitionedCallπ
StatefulPartitionedCallStatefulPartitionedCallxtextunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_12613066s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:????????? d:?????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:????????? d

_user_specified_nameX:MI
'
_output_shapes
:?????????

_user_specified_nametext

i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12614319

inputs
identity’
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
κS


H__inference_sequential_layer_call_and_return_conditional_losses_12612513
conv2d_input)
conv2d_12612452:@
conv2d_12612454:@,
conv2d_1_12612459:@ 
conv2d_1_12612461:	-
conv2d_2_12612466: 
conv2d_2_12612468:	-
conv2d_3_12612472: 
conv2d_3_12612474:	-
conv2d_4_12612479:+
batch_normalization_12612482:	+
batch_normalization_12612484:	+
batch_normalization_12612486:	+
batch_normalization_12612488:	-
conv2d_5_12612492:-
batch_normalization_1_12612495:	-
batch_normalization_1_12612497:	-
batch_normalization_1_12612499:	-
batch_normalization_1_12612501:	-
conv2d_6_12612506: 
conv2d_6_12612508:	
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’ conv2d_5/StatefulPartitionedCall’ conv2d_6/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_12612452conv2d_12612454*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? d@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_12611935ΰ
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? d@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_12611946η
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????2@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_12611751’
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_12612459conv2d_1_12612461*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_12611959η
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_12611970ξ
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12611763€
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_12612466conv2d_2_12612468*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_12611983η
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_12611994
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_12612472conv2d_3_12612474*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_12612006η
re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_12612017ξ
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12611775
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_4_12612479*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_12612027
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_12612482batch_normalization_12612484batch_normalization_12612486batch_normalization_12612488*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12611800ς
re_lu_4/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_12612045
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_5_12612492*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_12612054
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_1_12612495batch_normalization_1_12612497batch_normalization_1_12612499batch_normalization_1_12612501*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12611864τ
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_12612072ξ
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_12611915€
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_6_12612506conv2d_6_12612508*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_12612085η
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_6_layer_call_and_return_conditional_losses_12612096x
IdentityIdentity re_lu_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:] Y
/
_output_shapes
:????????? d
&
_user_specified_nameconv2d_input
Α
Σ
9__inference_vgg__feature_extractor_layer_call_fn_12613711
x!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	&

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	
identity’StatefulPartitionedCallγ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612626x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:????????? d

_user_specified_nameX
ά
 
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12611800

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ο
Ω
9__inference_vgg__feature_extractor_layer_call_fn_12612849
input_1!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	&

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	
identity’StatefulPartitionedCallε
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612761x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:????????? d
!
_user_specified_name	input_1
¨
Ή
F__inference_conv2d_5_layer_call_and_return_conditional_losses_12612054

inputs:
conv2d_readvariableop_resource:
identity’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
§
ΐ
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612894
input_1-
sequential_12612852:@!
sequential_12612854:@.
sequential_12612856:@"
sequential_12612858:	/
sequential_12612860:"
sequential_12612862:	/
sequential_12612864:"
sequential_12612866:	/
sequential_12612868:"
sequential_12612870:	"
sequential_12612872:	"
sequential_12612874:	"
sequential_12612876:	/
sequential_12612878:"
sequential_12612880:	"
sequential_12612882:	"
sequential_12612884:	"
sequential_12612886:	/
sequential_12612888:"
sequential_12612890:	
identity’"sequential/StatefulPartitionedCall©
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_12612852sequential_12612854sequential_12612856sequential_12612858sequential_12612860sequential_12612862sequential_12612864sequential_12612866sequential_12612868sequential_12612870sequential_12612872sequential_12612874sequential_12612876sequential_12612878sequential_12612880sequential_12612882sequential_12612884sequential_12612886sequential_12612888sequential_12612890* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_12612099
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????k
NoOpNoOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:X T
/
_output_shapes
:????????? d
!
_user_specified_name	input_1
μ
Ζ
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12611895

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ϋ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ί
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ΤS


H__inference_sequential_layer_call_and_return_conditional_losses_12612361

inputs)
conv2d_12612300:@
conv2d_12612302:@,
conv2d_1_12612307:@ 
conv2d_1_12612309:	-
conv2d_2_12612314: 
conv2d_2_12612316:	-
conv2d_3_12612320: 
conv2d_3_12612322:	-
conv2d_4_12612327:+
batch_normalization_12612330:	+
batch_normalization_12612332:	+
batch_normalization_12612334:	+
batch_normalization_12612336:	-
conv2d_5_12612340:-
batch_normalization_1_12612343:	-
batch_normalization_1_12612345:	-
batch_normalization_1_12612347:	-
batch_normalization_1_12612349:	-
conv2d_6_12612354: 
conv2d_6_12612356:	
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’ conv2d_5/StatefulPartitionedCall’ conv2d_6/StatefulPartitionedCallω
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12612300conv2d_12612302*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? d@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_12611935ΰ
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? d@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_12611946η
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????2@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_12611751’
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_12612307conv2d_1_12612309*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_12611959η
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_12611970ξ
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12611763€
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_12612314conv2d_2_12612316*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_12611983η
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_12611994
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_12612320conv2d_3_12612322*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_12612006η
re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_12612017ξ
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12611775
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_4_12612327*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_12612027
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_12612330batch_normalization_12612332batch_normalization_12612334batch_normalization_12612336*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12611831ς
re_lu_4/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_12612045
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_5_12612340*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_12612054
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_1_12612343batch_normalization_1_12612345batch_normalization_1_12612347batch_normalization_1_12612349*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12611895τ
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_12612072ξ
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_12611915€
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_6_12612354conv2d_6_12612356*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_12612085η
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_6_layer_call_and_return_conditional_losses_12612096x
IdentityIdentity re_lu_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:W S
/
_output_shapes
:????????? d
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_12611915

inputs
identity’
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ΨS


H__inference_sequential_layer_call_and_return_conditional_losses_12612099

inputs)
conv2d_12611936:@
conv2d_12611938:@,
conv2d_1_12611960:@ 
conv2d_1_12611962:	-
conv2d_2_12611984: 
conv2d_2_12611986:	-
conv2d_3_12612007: 
conv2d_3_12612009:	-
conv2d_4_12612028:+
batch_normalization_12612031:	+
batch_normalization_12612033:	+
batch_normalization_12612035:	+
batch_normalization_12612037:	-
conv2d_5_12612055:-
batch_normalization_1_12612058:	-
batch_normalization_1_12612060:	-
batch_normalization_1_12612062:	-
batch_normalization_1_12612064:	-
conv2d_6_12612086: 
conv2d_6_12612088:	
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’ conv2d_5/StatefulPartitionedCall’ conv2d_6/StatefulPartitionedCallω
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12611936conv2d_12611938*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? d@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_12611935ΰ
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? d@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_12611946η
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????2@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_12611751’
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_12611960conv2d_1_12611962*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_12611959η
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_12611970ξ
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12611763€
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_12611984conv2d_2_12611986*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_12611983η
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_12611994
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_12612007conv2d_3_12612009*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_12612006η
re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_12612017ξ
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12611775
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_4_12612028*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_12612027
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_12612031batch_normalization_12612033batch_normalization_12612035batch_normalization_12612037*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12611800ς
re_lu_4/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_12612045
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_5_12612055*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_12612054
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_1_12612058batch_normalization_1_12612060batch_normalization_1_12612062batch_normalization_1_12612064*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12611864τ
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_12612072ξ
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_12611915€
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_6_12612086conv2d_6_12612088*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_12612085η
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_re_lu_6_layer_call_and_return_conditional_losses_12612096x
IdentityIdentity re_lu_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:W S
/
_output_shapes
:????????? d
 
_user_specified_nameinputs
ν
a
E__inference_re_lu_3_layer_call_and_return_conditional_losses_12612017

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
¨
Ή
F__inference_conv2d_4_layer_call_and_return_conditional_losses_12612027

inputs:
conv2d_readvariableop_resource:
identity’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ζ
?
-__inference_sequential_layer_call_fn_12612449
conv2d_input!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	&

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	
identity’StatefulPartitionedCallή
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_12612361x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:????????? d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:????????? d
&
_user_specified_nameconv2d_input
΄


F__inference_conv2d_2_layer_call_and_return_conditional_losses_12611983

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
§

ύ
D__inference_conv2d_layer_call_and_return_conditional_losses_12614260

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? d@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? d@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? d
 
_user_specified_nameinputs"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*π
serving_defaultά
A
args_07
serving_default_args_0:0????????? d
9
args_1/
serving_default_args_1:0?????????@
output_14
StatefulPartitionedCall:0?????????tensorflow/serving/predict:€?

FeatureExtraction
AdaptiveAvgPool

Prediction
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures"
_tf_keras_model
Ζ
output_channel
ConvNet
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_model
₯
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
Ζ
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
20
21"
trackable_list_wrapper
¦
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
/11
012
113
414
515
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
2
(__inference_model_layer_call_fn_12613284
(__inference_model_layer_call_fn_12613334·
?²ͺ
FullArgSpec,
args$!
jself
jX
jtext

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Η2Δ
C__inference_model_layer_call_and_return_conditional_losses_12613474
C__inference_model_layer_call_and_return_conditional_losses_12613614·
?²ͺ
FullArgSpec,
args$!
jself
jX
jtext

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΥB?
#__inference__wrapped_model_12611742args_0args_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
,
;serving_default"
signature_map
 "
trackable_list_wrapper
’
<layer_with_weights-0
<layer-0
=layer-1
>layer-2
?layer_with_weights-1
?layer-3
@layer-4
Alayer-5
Blayer_with_weights-2
Blayer-6
Clayer-7
Dlayer_with_weights-3
Dlayer-8
Elayer-9
Flayer-10
Glayer_with_weights-4
Glayer-11
Hlayer_with_weights-5
Hlayer-12
Ilayer-13
Jlayer_with_weights-6
Jlayer-14
Klayer_with_weights-7
Klayer-15
Llayer-16
Mlayer-17
Nlayer_with_weights-8
Nlayer-18
Olayer-19
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_sequential
Ά
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519"
trackable_list_wrapper

"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
/11
012
113
414
515"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
‘2
9__inference_vgg__feature_extractor_layer_call_fn_12612669
9__inference_vgg__feature_extractor_layer_call_fn_12613711
9__inference_vgg__feature_extractor_layer_call_fn_12613756
9__inference_vgg__feature_extractor_layer_call_fn_12612849―
¦²’
FullArgSpec$
args
jself
jX

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12613835
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12613914
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612894
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612939―
¦²’
FullArgSpec$
args
jself
jX

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
λ2θ
=__inference_adaptive_average_pooling2d_layer_call_fn_12613919¦
²
FullArgSpec
args
jself
jinputs
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
X__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_12613954¦
²
FullArgSpec
args
jself
jinputs
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
%:#	2model/dense/kernel
:2model/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
?2Ο
(__inference_dense_layer_call_fn_12613963’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ν2κ
C__inference_dense_layer_call_and_return_conditional_losses_12613993’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
':%@2conv2d/kernel
:@2conv2d/bias
*:(@2conv2d_1/kernel
:2conv2d_1/bias
+:)2conv2d_2/kernel
:2conv2d_2/bias
+:)2conv2d_3/kernel
:2conv2d_3/bias
+:)2conv2d_4/kernel
(:&2batch_normalization/gamma
':%2batch_normalization/beta
0:. (2batch_normalization/moving_mean
4:2 (2#batch_normalization/moving_variance
+:)2conv2d_5/kernel
*:(2batch_normalization_1/gamma
):'2batch_normalization_1/beta
2:0 (2!batch_normalization_1/moving_mean
6:4 (2%batch_normalization_1/moving_variance
+:)2conv2d_6/kernel
:2conv2d_6/bias
<
-0
.1
22
33"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?BΟ
&__inference_signature_wrapper_12613666args_0args_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
»

"kernel
#bias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
»

$kernel
%bias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
¨
}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Α

&kernel
'bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Α

(kernel
)bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
«
‘	variables
’trainable_variables
£regularization_losses
€	keras_api
₯__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
·

*kernel
§	variables
¨trainable_variables
©regularization_losses
ͺ	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
ρ
	­axis
	+gamma
,beta
-moving_mean
.moving_variance
?	variables
―trainable_variables
°regularization_losses
±	keras_api
²__call__
+³&call_and_return_all_conditional_losses"
_tf_keras_layer
«
΄	variables
΅trainable_variables
Άregularization_losses
·	keras_api
Έ__call__
+Ή&call_and_return_all_conditional_losses"
_tf_keras_layer
·

/kernel
Ί	variables
»trainable_variables
Όregularization_losses
½	keras_api
Ύ__call__
+Ώ&call_and_return_all_conditional_losses"
_tf_keras_layer
ρ
	ΐaxis
	0gamma
1beta
2moving_mean
3moving_variance
Α	variables
Βtrainable_variables
Γregularization_losses
Δ	keras_api
Ε__call__
+Ζ&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Η	variables
Θtrainable_variables
Ιregularization_losses
Κ	keras_api
Λ__call__
+Μ&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ν	variables
Ξtrainable_variables
Οregularization_losses
Π	keras_api
Ρ__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
Α

4kernel
5bias
Σ	variables
Τtrainable_variables
Υregularization_losses
Φ	keras_api
Χ__call__
+Ψ&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ω	variables
Ϊtrainable_variables
Ϋregularization_losses
ά	keras_api
έ__call__
+ή&call_and_return_all_conditional_losses"
_tf_keras_layer
Ά
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519"
trackable_list_wrapper

"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
/11
012
113
414
515"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ίnon_trainable_variables
ΰlayers
αmetrics
 βlayer_regularization_losses
γlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
2?
-__inference_sequential_layer_call_fn_12612142
-__inference_sequential_layer_call_fn_12614038
-__inference_sequential_layer_call_fn_12614083
-__inference_sequential_layer_call_fn_12612449ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
H__inference_sequential_layer_call_and_return_conditional_losses_12614162
H__inference_sequential_layer_call_and_return_conditional_losses_12614241
H__inference_sequential_layer_call_and_return_conditional_losses_12612513
H__inference_sequential_layer_call_and_return_conditional_losses_12612577ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
<
-0
.1
22
33"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
δnon_trainable_variables
εlayers
ζmetrics
 ηlayer_regularization_losses
θlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
Σ2Π
)__inference_conv2d_layer_call_fn_12614250’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv2d_layer_call_and_return_conditional_losses_12614260’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ιnon_trainable_variables
κlayers
λmetrics
 μlayer_regularization_losses
νlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
?2Ο
(__inference_re_lu_layer_call_fn_12614265’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ν2κ
C__inference_re_lu_layer_call_and_return_conditional_losses_12614270’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ξnon_trainable_variables
οlayers
πmetrics
 ρlayer_regularization_losses
ςlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
Ϊ2Χ
0__inference_max_pooling2d_layer_call_fn_12614275’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
υ2ς
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_12614280’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
σnon_trainable_variables
τlayers
υmetrics
 φlayer_regularization_losses
χlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
Υ2?
+__inference_conv2d_1_layer_call_fn_12614289’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
π2ν
F__inference_conv2d_1_layer_call_and_return_conditional_losses_12614299’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ψnon_trainable_variables
ωlayers
ϊmetrics
 ϋlayer_regularization_losses
όlayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Τ2Ρ
*__inference_re_lu_1_layer_call_fn_12614304’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_1_layer_call_and_return_conditional_losses_12614309’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ύnon_trainable_variables
ώlayers
?metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ά2Ω
2__inference_max_pooling2d_1_layer_call_fn_12614314’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χ2τ
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12614319’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Υ2?
+__inference_conv2d_2_layer_call_fn_12614328’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
π2ν
F__inference_conv2d_2_layer_call_and_return_conditional_losses_12614338’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Τ2Ρ
*__inference_re_lu_2_layer_call_fn_12614343’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_2_layer_call_and_return_conditional_losses_12614348’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Υ2?
+__inference_conv2d_3_layer_call_fn_12614357’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
π2ν
F__inference_conv2d_3_layer_call_and_return_conditional_losses_12614367’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
Τ2Ρ
*__inference_re_lu_3_layer_call_fn_12614372’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_3_layer_call_and_return_conditional_losses_12614377’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
‘	variables
’trainable_variables
£regularization_losses
₯__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
ά2Ω
2__inference_max_pooling2d_2_layer_call_fn_12614382’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χ2τ
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12614387’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
Υ2?
+__inference_conv2d_4_layer_call_fn_12614394’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
π2ν
F__inference_conv2d_4_layer_call_and_return_conditional_losses_12614401’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
<
+0
,1
-2
.3"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 non_trainable_variables
‘layers
’metrics
 £layer_regularization_losses
€layer_metrics
?	variables
―trainable_variables
°regularization_losses
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
ͺ2§
6__inference_batch_normalization_layer_call_fn_12614414
6__inference_batch_normalization_layer_call_fn_12614427΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΰ2έ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12614445
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12614463΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
₯non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
΄	variables
΅trainable_variables
Άregularization_losses
Έ__call__
+Ή&call_and_return_all_conditional_losses
'Ή"call_and_return_conditional_losses"
_generic_user_object
Τ2Ρ
*__inference_re_lu_4_layer_call_fn_12614468’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_4_layer_call_and_return_conditional_losses_12614473’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
'
/0"
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ͺnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
?layer_metrics
Ί	variables
»trainable_variables
Όregularization_losses
Ύ__call__
+Ώ&call_and_return_all_conditional_losses
'Ώ"call_and_return_conditional_losses"
_generic_user_object
Υ2?
+__inference_conv2d_5_layer_call_fn_12614480’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
π2ν
F__inference_conv2d_5_layer_call_and_return_conditional_losses_12614487’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
―non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
Α	variables
Βtrainable_variables
Γregularization_losses
Ε__call__
+Ζ&call_and_return_all_conditional_losses
'Ζ"call_and_return_conditional_losses"
_generic_user_object
?2«
8__inference_batch_normalization_1_layer_call_fn_12614500
8__inference_batch_normalization_1_layer_call_fn_12614513΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
δ2α
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12614531
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12614549΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
΄non_trainable_variables
΅layers
Άmetrics
 ·layer_regularization_losses
Έlayer_metrics
Η	variables
Θtrainable_variables
Ιregularization_losses
Λ__call__
+Μ&call_and_return_all_conditional_losses
'Μ"call_and_return_conditional_losses"
_generic_user_object
Τ2Ρ
*__inference_re_lu_5_layer_call_fn_12614554’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_5_layer_call_and_return_conditional_losses_12614559’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ήnon_trainable_variables
Ίlayers
»metrics
 Όlayer_regularization_losses
½layer_metrics
Ν	variables
Ξtrainable_variables
Οregularization_losses
Ρ__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
ά2Ω
2__inference_max_pooling2d_3_layer_call_fn_12614564’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χ2τ
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_12614569’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ύnon_trainable_variables
Ώlayers
ΐmetrics
 Αlayer_regularization_losses
Βlayer_metrics
Σ	variables
Τtrainable_variables
Υregularization_losses
Χ__call__
+Ψ&call_and_return_all_conditional_losses
'Ψ"call_and_return_conditional_losses"
_generic_user_object
Υ2?
+__inference_conv2d_6_layer_call_fn_12614578’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
π2ν
F__inference_conv2d_6_layer_call_and_return_conditional_losses_12614588’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Γnon_trainable_variables
Δlayers
Εmetrics
 Ζlayer_regularization_losses
Ηlayer_metrics
Ω	variables
Ϊtrainable_variables
Ϋregularization_losses
έ__call__
+ή&call_and_return_all_conditional_losses
'ή"call_and_return_conditional_losses"
_generic_user_object
Τ2Ρ
*__inference_re_lu_6_layer_call_fn_12614593’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_6_layer_call_and_return_conditional_losses_12614598’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
<
-0
.1
22
33"
trackable_list_wrapper
Ά
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
N18
O19"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapperΤ
#__inference__wrapped_model_12611742¬"#$%&'()*+,-./012345Y’V
O’L
(%
args_0????????? d
 
args_1?????????
ͺ "7ͺ4
2
output_1&#
output_1?????????Ζ
X__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_12613954j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
=__inference_adaptive_average_pooling2d_layer_call_fn_12613919]8’5
.’+
)&
inputs?????????
ͺ "!?????????π
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_126145310123N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 π
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_126145490123N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 Θ
8__inference_batch_normalization_1_layer_call_fn_126145000123N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????Θ
8__inference_batch_normalization_1_layer_call_fn_126145130123N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????ξ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12614445+,-.N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 ξ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_12614463+,-.N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 Ζ
6__inference_batch_normalization_layer_call_fn_12614414+,-.N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????Ζ
6__inference_batch_normalization_layer_call_fn_12614427+,-.N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????·
F__inference_conv2d_1_layer_call_and_return_conditional_losses_12614299m$%7’4
-’*
(%
inputs?????????2@
ͺ ".’+
$!
0?????????2
 
+__inference_conv2d_1_layer_call_fn_12614289`$%7’4
-’*
(%
inputs?????????2@
ͺ "!?????????2Έ
F__inference_conv2d_2_layer_call_and_return_conditional_losses_12614338n&'8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
+__inference_conv2d_2_layer_call_fn_12614328a&'8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
F__inference_conv2d_3_layer_call_and_return_conditional_losses_12614367n()8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
+__inference_conv2d_3_layer_call_fn_12614357a()8’5
.’+
)&
inputs?????????
ͺ "!?????????·
F__inference_conv2d_4_layer_call_and_return_conditional_losses_12614401m*8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
+__inference_conv2d_4_layer_call_fn_12614394`*8’5
.’+
)&
inputs?????????
ͺ "!?????????·
F__inference_conv2d_5_layer_call_and_return_conditional_losses_12614487m/8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
+__inference_conv2d_5_layer_call_fn_12614480`/8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
F__inference_conv2d_6_layer_call_and_return_conditional_losses_12614588n458’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
+__inference_conv2d_6_layer_call_fn_12614578a458’5
.’+
)&
inputs?????????
ͺ "!?????????΄
D__inference_conv2d_layer_call_and_return_conditional_losses_12614260l"#7’4
-’*
(%
inputs????????? d
ͺ "-’*
# 
0????????? d@
 
)__inference_conv2d_layer_call_fn_12614250_"#7’4
-’*
(%
inputs????????? d
ͺ " ????????? d@¬
C__inference_dense_layer_call_and_return_conditional_losses_12613993e4’1
*’'
%"
inputs?????????
ͺ ")’&

0?????????
 
(__inference_dense_layer_call_fn_12613963X4’1
*’'
%"
inputs?????????
ͺ "?????????π
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12614319R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Θ
2__inference_max_pooling2d_1_layer_call_fn_12614314R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????π
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12614387R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Θ
2__inference_max_pooling2d_2_layer_call_fn_12614382R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????π
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_12614569R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Θ
2__inference_max_pooling2d_3_layer_call_fn_12614564R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????ξ
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_12614280R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ζ
0__inference_max_pooling2d_layer_call_fn_12614275R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????γ
C__inference_model_layer_call_and_return_conditional_losses_12613474"#$%&'()*+,-./012345V’S
L’I
# 
X????????? d

text?????????
p 
ͺ ")’&

0?????????
 γ
C__inference_model_layer_call_and_return_conditional_losses_12613614"#$%&'()*+,-./012345V’S
L’I
# 
X????????? d

text?????????
p
ͺ ")’&

0?????????
 »
(__inference_model_layer_call_fn_12613284"#$%&'()*+,-./012345V’S
L’I
# 
X????????? d

text?????????
p 
ͺ "?????????»
(__inference_model_layer_call_fn_12613334"#$%&'()*+,-./012345V’S
L’I
# 
X????????? d

text?????????
p
ͺ "?????????³
E__inference_re_lu_1_layer_call_and_return_conditional_losses_12614309j8’5
.’+
)&
inputs?????????2
ͺ ".’+
$!
0?????????2
 
*__inference_re_lu_1_layer_call_fn_12614304]8’5
.’+
)&
inputs?????????2
ͺ "!?????????2³
E__inference_re_lu_2_layer_call_and_return_conditional_losses_12614348j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
*__inference_re_lu_2_layer_call_fn_12614343]8’5
.’+
)&
inputs?????????
ͺ "!?????????³
E__inference_re_lu_3_layer_call_and_return_conditional_losses_12614377j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
*__inference_re_lu_3_layer_call_fn_12614372]8’5
.’+
)&
inputs?????????
ͺ "!?????????³
E__inference_re_lu_4_layer_call_and_return_conditional_losses_12614473j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
*__inference_re_lu_4_layer_call_fn_12614468]8’5
.’+
)&
inputs?????????
ͺ "!?????????³
E__inference_re_lu_5_layer_call_and_return_conditional_losses_12614559j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
*__inference_re_lu_5_layer_call_fn_12614554]8’5
.’+
)&
inputs?????????
ͺ "!?????????³
E__inference_re_lu_6_layer_call_and_return_conditional_losses_12614598j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
*__inference_re_lu_6_layer_call_fn_12614593]8’5
.’+
)&
inputs?????????
ͺ "!?????????―
C__inference_re_lu_layer_call_and_return_conditional_losses_12614270h7’4
-’*
(%
inputs????????? d@
ͺ "-’*
# 
0????????? d@
 
(__inference_re_lu_layer_call_fn_12614265[7’4
-’*
(%
inputs????????? d@
ͺ " ????????? d@Ϊ
H__inference_sequential_layer_call_and_return_conditional_losses_12612513"#$%&'()*+,-./012345E’B
;’8
.+
conv2d_input????????? d
p 

 
ͺ ".’+
$!
0?????????
 Ϊ
H__inference_sequential_layer_call_and_return_conditional_losses_12612577"#$%&'()*+,-./012345E’B
;’8
.+
conv2d_input????????? d
p

 
ͺ ".’+
$!
0?????????
 Τ
H__inference_sequential_layer_call_and_return_conditional_losses_12614162"#$%&'()*+,-./012345?’<
5’2
(%
inputs????????? d
p 

 
ͺ ".’+
$!
0?????????
 Τ
H__inference_sequential_layer_call_and_return_conditional_losses_12614241"#$%&'()*+,-./012345?’<
5’2
(%
inputs????????? d
p

 
ͺ ".’+
$!
0?????????
 ²
-__inference_sequential_layer_call_fn_12612142"#$%&'()*+,-./012345E’B
;’8
.+
conv2d_input????????? d
p 

 
ͺ "!?????????²
-__inference_sequential_layer_call_fn_12612449"#$%&'()*+,-./012345E’B
;’8
.+
conv2d_input????????? d
p

 
ͺ "!?????????«
-__inference_sequential_layer_call_fn_12614038z"#$%&'()*+,-./012345?’<
5’2
(%
inputs????????? d
p 

 
ͺ "!?????????«
-__inference_sequential_layer_call_fn_12614083z"#$%&'()*+,-./012345?’<
5’2
(%
inputs????????? d
p

 
ͺ "!?????????λ
&__inference_signature_wrapper_12613666ΐ"#$%&'()*+,-./012345m’j
’ 
cͺ`
2
args_0(%
args_0????????? d
*
args_1 
args_1?????????"7ͺ4
2
output_1&#
output_1?????????έ
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612894"#$%&'()*+,-./012345<’9
2’/
)&
input_1????????? d
p 
ͺ ".’+
$!
0?????????
 έ
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12612939"#$%&'()*+,-./012345<’9
2’/
)&
input_1????????? d
p
ͺ ".’+
$!
0?????????
 Φ
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12613835~"#$%&'()*+,-./0123456’3
,’)
# 
X????????? d
p 
ͺ ".’+
$!
0?????????
 Φ
T__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_12613914~"#$%&'()*+,-./0123456’3
,’)
# 
X????????? d
p
ͺ ".’+
$!
0?????????
 ΄
9__inference_vgg__feature_extractor_layer_call_fn_12612669w"#$%&'()*+,-./012345<’9
2’/
)&
input_1????????? d
p 
ͺ "!?????????΄
9__inference_vgg__feature_extractor_layer_call_fn_12612849w"#$%&'()*+,-./012345<’9
2’/
)&
input_1????????? d
p
ͺ "!??????????
9__inference_vgg__feature_extractor_layer_call_fn_12613711q"#$%&'()*+,-./0123456’3
,’)
# 
X????????? d
p 
ͺ "!??????????
9__inference_vgg__feature_extractor_layer_call_fn_12613756q"#$%&'()*+,-./0123456’3
,’)
# 
X????????? d
p
ͺ "!?????????