??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
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
?
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
?
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
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
?
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
?
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
?
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
delete_old_dirsbool(?
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
?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ??
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.22v2.8.2-0-g2ea19cbb5758??
?
model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_namemodel/dense/kernel
z
&model/dense/kernel/Read/ReadVariableOpReadVariableOpmodel/dense/kernel*
_output_shapes
:	?*
dtype0
x
model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namemodel/dense/bias
q
$model/dense/bias/Read/ReadVariableOpReadVariableOpmodel/dense/bias*
_output_shapes
:*
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
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:@?*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:?*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:??*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:?*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_3/kernel
}
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:??*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:?*
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:??*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:?*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_5/kernel
}
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*(
_output_shapes
:??*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:?*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_6/kernel
}
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*(
_output_shapes
:??*
dtype0
s
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_6/bias
l
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
FeatureExtraction
AdaptiveAvgPool
flatten

Prediction
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?
output_channel
ConvNet
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 

	keras_api* 
?

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
?
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719
20
21*
?
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
111
212
313
614
715
16
17*
* 
?
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

=serving_default* 
* 
?
>layer_with_weights-0
>layer-0
?layer-1
@layer-2
Alayer_with_weights-1
Alayer-3
Blayer-4
Clayer-5
Dlayer_with_weights-2
Dlayer-6
Elayer-7
Flayer_with_weights-3
Flayer-8
Glayer-9
Hlayer-10
Ilayer_with_weights-4
Ilayer-11
Jlayer_with_weights-5
Jlayer-12
Klayer-13
Llayer_with_weights-6
Llayer-14
Mlayer_with_weights-7
Mlayer-15
Nlayer-16
Olayer-17
Player_with_weights-8
Player-18
Qlayer-19
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses*
?
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719*
z
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
111
212
313
614
715*
* 
?
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
* 
XR
VARIABLE_VALUEmodel/dense/kernel,Prediction/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEmodel/dense/bias*Prediction/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
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
 
/0
01
42
53*
 
0
1
2
3*
* 
* 
* 
* 
?

$kernel
%bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses*
?
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses* 
?
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses* 
?

&kernel
'bias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses*
?
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

(kernel
)bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

*kernel
+bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

,kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis
	-gamma
.beta
/moving_mean
0moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

1kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis
	2gamma
3beta
4moving_mean
5moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

6kernel
7bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719*
z
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
111
212
313
614
715*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*
* 
* 
 
/0
01
42
53*

0*
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
$0
%1*

$0
%1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses* 
* 
* 

&0
'1*

&0
'1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

(0
)1*

(0
)1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

*0
+1*

*0
+1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

,0*

,0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
 
-0
.1
/2
03*

-0
.1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

10*

10*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
 
20
31
42
53*

20
31*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

60
71*

60
71*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
 
/0
01
42
53*
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19*
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
/0
01*
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
40
51*
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
?
serving_default_args_0Placeholder*0
_output_shapes
:?????????@?*
dtype0*%
shape:?????????@?
y
serving_default_args_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_5/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_6/kernelconv2d_6/biasmodel/dense/kernelmodel/dense/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????1*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *0
f+R)
'__inference_signature_wrapper_345048565
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
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
GPU2*0J 8? *+
f&R$
"__inference__traced_save_345049612
?
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
GPU2*0J 8? *.
f)R'
%__inference__traced_restore_345049688ڦ
?
?
G__inference_conv2d_4_layer_call_and_return_conditional_losses_345049325

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????2?^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????2?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
b
F__inference_re_lu_5_layer_call_and_return_conditional_losses_345049483

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????2?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?

?
G__inference_conv2d_1_layer_call_and_return_conditional_losses_345049223

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:????????? d?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? d@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? d@
 
_user_specified_nameinputs
?
?
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047585
x.
sequential_345047543:@"
sequential_345047545:@/
sequential_345047547:@?#
sequential_345047549:	?0
sequential_345047551:??#
sequential_345047553:	?0
sequential_345047555:??#
sequential_345047557:	?0
sequential_345047559:??#
sequential_345047561:	?#
sequential_345047563:	?#
sequential_345047565:	?#
sequential_345047567:	?0
sequential_345047569:??#
sequential_345047571:	?#
sequential_345047573:	?#
sequential_345047575:	?#
sequential_345047577:	?0
sequential_345047579:??#
sequential_345047581:	?
identity??"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_345047543sequential_345047545sequential_345047547sequential_345047549sequential_345047551sequential_345047553sequential_345047555sequential_345047557sequential_345047559sequential_345047561sequential_345047563sequential_345047565sequential_345047567sequential_345047569sequential_345047571sequential_345047573sequential_345047575sequential_345047577sequential_345047579sequential_345047581* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_345047185?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?k
NoOpNoOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:S O
0
_output_shapes
:?????????@?

_user_specified_nameX
?
j
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_345049493

inputs
identity?
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
?
O
3__inference_max_pooling2d_1_layer_call_fn_345049238

inputs
identity?
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
GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_345046587?
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
?
?
'__inference_signature_wrapper_345048565

args_0

args_1!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?&

unknown_12:??

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:
identity??StatefulPartitionedCall?
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
:?????????1*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__wrapped_model_345046566s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????@?:?????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@?
 
_user_specified_nameargs_0:OK
'
_output_shapes
:?????????
 
_user_specified_nameargs_1
?"
?

D__inference_model_layer_call_and_return_conditional_losses_345048036
x
text:
 vgg__feature_extractor_345047985:@.
 vgg__feature_extractor_345047987:@;
 vgg__feature_extractor_345047989:@?/
 vgg__feature_extractor_345047991:	?<
 vgg__feature_extractor_345047993:??/
 vgg__feature_extractor_345047995:	?<
 vgg__feature_extractor_345047997:??/
 vgg__feature_extractor_345047999:	?<
 vgg__feature_extractor_345048001:??/
 vgg__feature_extractor_345048003:	?/
 vgg__feature_extractor_345048005:	?/
 vgg__feature_extractor_345048007:	?/
 vgg__feature_extractor_345048009:	?<
 vgg__feature_extractor_345048011:??/
 vgg__feature_extractor_345048013:	?/
 vgg__feature_extractor_345048015:	?/
 vgg__feature_extractor_345048017:	?/
 vgg__feature_extractor_345048019:	?<
 vgg__feature_extractor_345048021:??/
 vgg__feature_extractor_345048023:	?"
dense_345048030:	?
dense_345048032:
identity??dense/StatefulPartitionedCall?.vgg__feature_extractor/StatefulPartitionedCall?
.vgg__feature_extractor/StatefulPartitionedCallStatefulPartitionedCallx vgg__feature_extractor_345047985 vgg__feature_extractor_345047987 vgg__feature_extractor_345047989 vgg__feature_extractor_345047991 vgg__feature_extractor_345047993 vgg__feature_extractor_345047995 vgg__feature_extractor_345047997 vgg__feature_extractor_345047999 vgg__feature_extractor_345048001 vgg__feature_extractor_345048003 vgg__feature_extractor_345048005 vgg__feature_extractor_345048007 vgg__feature_extractor_345048009 vgg__feature_extractor_345048011 vgg__feature_extractor_345048013 vgg__feature_extractor_345048015 vgg__feature_extractor_345048017 vgg__feature_extractor_345048019 vgg__feature_extractor_345048021 vgg__feature_extractor_345048023* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047585g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
	transpose	Transpose7vgg__feature_extractor/StatefulPartitionedCall:output:0transpose/perm:output:0*
T0*0
_output_shapes
:?????????1??
*adaptive_average_pooling2d/PartitionedCallPartitionedCalltranspose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *b
f]R[
Y__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_345047875?
SqueezeSqueeze3adaptive_average_pooling2d/PartitionedCall:output:0*
T0*,
_output_shapes
:?????????1?*
squeeze_dims
?
dense/StatefulPartitionedCallStatefulPartitionedCallSqueeze:output:0dense_345048030dense_345048032*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_345047908y
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????1?
NoOpNoOp^dense/StatefulPartitionedCall/^vgg__feature_extractor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????@?:?????????: : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.vgg__feature_extractor/StatefulPartitionedCall.vgg__feature_extractor/StatefulPartitionedCall:S O
0
_output_shapes
:?????????@?

_user_specified_nameX:MI
'
_output_shapes
:?????????

_user_specified_nametext
?W
?
%__inference__traced_restore_345049688
file_prefix6
#assignvariableop_model_dense_kernel:	?1
#assignvariableop_1_model_dense_bias::
 assignvariableop_2_conv2d_kernel:@,
assignvariableop_3_conv2d_bias:@=
"assignvariableop_4_conv2d_1_kernel:@?/
 assignvariableop_5_conv2d_1_bias:	?>
"assignvariableop_6_conv2d_2_kernel:??/
 assignvariableop_7_conv2d_2_bias:	?>
"assignvariableop_8_conv2d_3_kernel:??/
 assignvariableop_9_conv2d_3_bias:	??
#assignvariableop_10_conv2d_4_kernel:??<
-assignvariableop_11_batch_normalization_gamma:	?;
,assignvariableop_12_batch_normalization_beta:	?B
3assignvariableop_13_batch_normalization_moving_mean:	?F
7assignvariableop_14_batch_normalization_moving_variance:	??
#assignvariableop_15_conv2d_5_kernel:??>
/assignvariableop_16_batch_normalization_1_gamma:	?=
.assignvariableop_17_batch_normalization_1_beta:	?D
5assignvariableop_18_batch_normalization_1_moving_mean:	?H
9assignvariableop_19_batch_normalization_1_moving_variance:	??
#assignvariableop_20_conv2d_6_kernel:??0
!assignvariableop_21_conv2d_6_bias:	?
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B,Prediction/kernel/.ATTRIBUTES/VARIABLE_VALUEB*Prediction/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp#assignvariableop_model_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp#assignvariableop_1_model_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_batch_normalization_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp,assignvariableop_12_batch_normalization_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp3assignvariableop_13_batch_normalization_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_batch_normalization_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv2d_5_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_1_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_1_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_1_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_1_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv2d_6_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv2d_6_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: ?
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
?
b
F__inference_re_lu_1_layer_call_and_return_conditional_losses_345046794

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:????????? d?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:????????? d?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:????????? d?:X T
0
_output_shapes
:????????? d?
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_345049204

inputs
identity?
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
?	
?
7__inference_batch_normalization_layer_call_fn_345049338

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345046624?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_re_lu_4_layer_call_and_return_conditional_losses_345046869

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????2?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?

?
E__inference_conv2d_layer_call_and_return_conditional_losses_345046759

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????@?@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
b
F__inference_re_lu_5_layer_call_and_return_conditional_losses_345046896

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????2?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_345046739

inputs
identity?
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
?

?
G__inference_conv2d_3_layer_call_and_return_conditional_losses_345046830

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????2?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????2?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_345046599

inputs
identity?
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
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345046655

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?a
?
I__inference_sequential_layer_call_and_return_conditional_losses_345049086

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@B
'conv2d_1_conv2d_readvariableop_resource:@?7
(conv2d_1_biasadd_readvariableop_resource:	?C
'conv2d_2_conv2d_readvariableop_resource:??7
(conv2d_2_biasadd_readvariableop_resource:	?C
'conv2d_3_conv2d_readvariableop_resource:??7
(conv2d_3_biasadd_readvariableop_resource:	?C
'conv2d_4_conv2d_readvariableop_resource:??:
+batch_normalization_readvariableop_resource:	?<
-batch_normalization_readvariableop_1_resource:	?K
<batch_normalization_fusedbatchnormv3_readvariableop_resource:	?M
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	?C
'conv2d_5_conv2d_readvariableop_resource:??<
-batch_normalization_1_readvariableop_resource:	?>
/batch_normalization_1_readvariableop_1_resource:	?M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	?C
'conv2d_6_conv2d_readvariableop_resource:??7
(conv2d_6_biasadd_readvariableop_resource:	?
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@f

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?@?
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:????????? d@*
ksize
*
paddingVALID*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?j
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:????????? d??
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?j
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_3/Conv2DConv2Dre_lu_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?j
re_lu_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
max_pooling2d_2/MaxPoolMaxPoolre_lu_3/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
is_training( y
re_lu_4/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_5/Conv2DConv2Dre_lu_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
is_training( {
re_lu_5/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
max_pooling2d_3/MaxPoolMaxPoolre_lu_5/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_6/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?*
paddingVALID*
strides
?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?j
re_lu_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????1?r
IdentityIdentityre_lu_6/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????1??
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 2j
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
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
u
Y__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_345047875

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0inputs*
T0*?

_output_shapes?

?
:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*
	num_split1?
stackPacksplit:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7split:output:8split:output:9split:output:10split:output:11split:output:12split:output:13split:output:14split:output:15split:output:16split:output:17split:output:18split:output:19split:output:20split:output:21split:output:22split:output:23split:output:24split:output:25split:output:26split:output:27split:output:28split:output:29split:output:30split:output:31split:output:32split:output:33split:output:34split:output:35split:output:36split:output:37split:output:38split:output:39split:output:40split:output:41split:output:42split:output:43split:output:44split:output:45split:output:46split:output:47split:output:48*
N1*
T0*4
_output_shapes"
 :?????????1?*

axisS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0stack:output:0*
T0*4
_output_shapes"
 :?????????1?*
	num_splity
stack_1Packsplit_1:output:0*
N*
T0*8
_output_shapes&
$:"?????????1?*

axisg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      z
MeanMeanstack_1:output:0Mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????1?^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:?????????1?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????1?:X T
0
_output_shapes
:?????????1?
 
_user_specified_nameinputs
?
?
D__inference_dense_layer_call_and_return_conditional_losses_345047908

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
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
value	B : ?
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
value	B : ?
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
value	B : ?
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
:?????????1??
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????1?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:?????????1?
 
_user_specified_nameinputs
?T
?

I__inference_sequential_layer_call_and_return_conditional_losses_345046923

inputs*
conv2d_345046760:@
conv2d_345046762:@-
conv2d_1_345046784:@?!
conv2d_1_345046786:	?.
conv2d_2_345046808:??!
conv2d_2_345046810:	?.
conv2d_3_345046831:??!
conv2d_3_345046833:	?.
conv2d_4_345046852:??,
batch_normalization_345046855:	?,
batch_normalization_345046857:	?,
batch_normalization_345046859:	?,
batch_normalization_345046861:	?.
conv2d_5_345046879:??.
batch_normalization_1_345046882:	?.
batch_normalization_1_345046884:	?.
batch_normalization_1_345046886:	?.
batch_normalization_1_345046888:	?.
conv2d_6_345046910:??!
conv2d_6_345046912:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_345046760conv2d_345046762*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_345046759?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_345046770?
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
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
GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_345046575?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_345046784conv2d_1_345046786*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? d?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_345046783?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? d?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_345046794?
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_345046587?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_345046808conv2d_2_345046810*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_345046807?
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_345046818?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_345046831conv2d_3_345046833*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_345046830?
re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_345046841?
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_345046599?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_4_345046852*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_345046851?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_345046855batch_normalization_345046857batch_normalization_345046859batch_normalization_345046861*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345046624?
re_lu_4/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_345046869?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_5_345046879*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_345046878?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_1_345046882batch_normalization_1_345046884batch_normalization_1_345046886batch_normalization_1_345046888*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345046688?
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_345046896?
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_345046739?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_6_345046910conv2d_6_345046912*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_345046909?
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_345046920x
IdentityIdentity re_lu_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1??
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:X T
0
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
?
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047450
x.
sequential_345047408:@"
sequential_345047410:@/
sequential_345047412:@?#
sequential_345047414:	?0
sequential_345047416:??#
sequential_345047418:	?0
sequential_345047420:??#
sequential_345047422:	?0
sequential_345047424:??#
sequential_345047426:	?#
sequential_345047428:	?#
sequential_345047430:	?#
sequential_345047432:	?0
sequential_345047434:??#
sequential_345047436:	?#
sequential_345047438:	?#
sequential_345047440:	?#
sequential_345047442:	?0
sequential_345047444:??#
sequential_345047446:	?
identity??"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_345047408sequential_345047410sequential_345047412sequential_345047414sequential_345047416sequential_345047418sequential_345047420sequential_345047422sequential_345047424sequential_345047426sequential_345047428sequential_345047430sequential_345047432sequential_345047434sequential_345047436sequential_345047438sequential_345047440sequential_345047442sequential_345047444sequential_345047446* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_345046923?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?k
NoOpNoOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:S O
0
_output_shapes
:?????????@?

_user_specified_nameX
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345049455

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_re_lu_layer_call_and_return_conditional_losses_345046770

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????@?@c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????@?@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?@:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
?
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047718
input_1.
sequential_345047676:@"
sequential_345047678:@/
sequential_345047680:@?#
sequential_345047682:	?0
sequential_345047684:??#
sequential_345047686:	?0
sequential_345047688:??#
sequential_345047690:	?0
sequential_345047692:??#
sequential_345047694:	?#
sequential_345047696:	?#
sequential_345047698:	?#
sequential_345047700:	?0
sequential_345047702:??#
sequential_345047704:	?#
sequential_345047706:	?#
sequential_345047708:	?#
sequential_345047710:	?0
sequential_345047712:??#
sequential_345047714:	?
identity??"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_345047676sequential_345047678sequential_345047680sequential_345047682sequential_345047684sequential_345047686sequential_345047688sequential_345047690sequential_345047692sequential_345047694sequential_345047696sequential_345047698sequential_345047700sequential_345047702sequential_345047704sequential_345047706sequential_345047708sequential_345047710sequential_345047712sequential_345047714* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_345046923?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?k
NoOpNoOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?
?
,__inference_conv2d_1_layer_call_fn_345049213

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? d?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_345046783x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:????????? d?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? d@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? d@
 
_user_specified_nameinputs
?
?
G__inference_conv2d_4_layer_call_and_return_conditional_losses_345046851

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????2?^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????2?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?T
?

I__inference_sequential_layer_call_and_return_conditional_losses_345047401
conv2d_input*
conv2d_345047340:@
conv2d_345047342:@-
conv2d_1_345047347:@?!
conv2d_1_345047349:	?.
conv2d_2_345047354:??!
conv2d_2_345047356:	?.
conv2d_3_345047360:??!
conv2d_3_345047362:	?.
conv2d_4_345047367:??,
batch_normalization_345047370:	?,
batch_normalization_345047372:	?,
batch_normalization_345047374:	?,
batch_normalization_345047376:	?.
conv2d_5_345047380:??.
batch_normalization_1_345047383:	?.
batch_normalization_1_345047385:	?.
batch_normalization_1_345047387:	?.
batch_normalization_1_345047389:	?.
conv2d_6_345047394:??!
conv2d_6_345047396:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_345047340conv2d_345047342*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_345046759?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_345046770?
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
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
GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_345046575?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_345047347conv2d_1_345047349*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? d?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_345046783?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? d?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_345046794?
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_345046587?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_345047354conv2d_2_345047356*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_345046807?
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_345046818?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_345047360conv2d_3_345047362*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_345046830?
re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_345046841?
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_345046599?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_4_345047367*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_345046851?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_345047370batch_normalization_345047372batch_normalization_345047374batch_normalization_345047376*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345046655?
re_lu_4/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_345046869?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_5_345047380*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_345046878?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_1_345047383batch_normalization_1_345047385batch_normalization_1_345047387batch_normalization_1_345047389*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345046719?
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_345046896?
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_345046739?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_6_345047394conv2d_6_345047396*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_345046909?
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_345046920x
IdentityIdentity re_lu_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1??
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:^ Z
0
_output_shapes
:?????????@?
&
_user_specified_nameconv2d_input
?
b
F__inference_re_lu_2_layer_call_and_return_conditional_losses_345046818

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????2?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
?
:__inference_vgg__feature_extractor_layer_call_fn_345048610
x!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?&

unknown_12:??

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?
identity??StatefulPartitionedCall?
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
:?????????1?*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047450x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:?????????@?

_user_specified_nameX
?s
?
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345048734
xJ
0sequential_conv2d_conv2d_readvariableop_resource:@?
1sequential_conv2d_biasadd_readvariableop_resource:@M
2sequential_conv2d_1_conv2d_readvariableop_resource:@?B
3sequential_conv2d_1_biasadd_readvariableop_resource:	?N
2sequential_conv2d_2_conv2d_readvariableop_resource:??B
3sequential_conv2d_2_biasadd_readvariableop_resource:	?N
2sequential_conv2d_3_conv2d_readvariableop_resource:??B
3sequential_conv2d_3_biasadd_readvariableop_resource:	?N
2sequential_conv2d_4_conv2d_readvariableop_resource:??E
6sequential_batch_normalization_readvariableop_resource:	?G
8sequential_batch_normalization_readvariableop_1_resource:	?V
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:	?X
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	?N
2sequential_conv2d_5_conv2d_readvariableop_resource:??G
8sequential_batch_normalization_1_readvariableop_resource:	?I
:sequential_batch_normalization_1_readvariableop_1_resource:	?X
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	?Z
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	?N
2sequential_conv2d_6_conv2d_readvariableop_resource:??B
3sequential_conv2d_6_biasadd_readvariableop_resource:	?
identity??>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?/sequential/batch_normalization_1/ReadVariableOp?1sequential/batch_normalization_1/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?*sequential/conv2d_3/BiasAdd/ReadVariableOp?)sequential/conv2d_3/Conv2D/ReadVariableOp?)sequential/conv2d_4/Conv2D/ReadVariableOp?)sequential/conv2d_5/Conv2D/ReadVariableOp?*sequential/conv2d_6/BiasAdd/ReadVariableOp?)sequential/conv2d_6/Conv2D/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@|
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?@?
 sequential/max_pooling2d/MaxPoolMaxPool#sequential/re_lu/Relu:activations:0*/
_output_shapes
:????????? d@*
ksize
*
paddingVALID*
strides
?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?*
paddingSAME*
strides
?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d??
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:????????? d??
"sequential/max_pooling2d_1/MaxPoolMaxPool%sequential/re_lu_1/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2??
sequential/re_lu_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential/conv2d_3/Conv2DConv2D%sequential/re_lu_2/Relu:activations:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
*sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/conv2d_3/BiasAddBiasAdd#sequential/conv2d_3/Conv2D:output:02sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2??
sequential/re_lu_3/ReluRelu$sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
"sequential/max_pooling2d_2/MaxPoolMaxPool%sequential/re_lu_3/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
)sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential/conv2d_4/Conv2DConv2D+sequential/max_pooling2d_2/MaxPool:output:01sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3#sequential/conv2d_4/Conv2D:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
is_training( ?
sequential/re_lu_4/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
)sequential/conv2d_5/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential/conv2d_5/Conv2DConv2D%sequential/re_lu_4/Relu:activations:01sequential/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#sequential/conv2d_5/Conv2D:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
is_training( ?
sequential/re_lu_5/ReluRelu5sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
"sequential/max_pooling2d_3/MaxPoolMaxPool%sequential/re_lu_5/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
)sequential/conv2d_6/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential/conv2d_6/Conv2DConv2D+sequential/max_pooling2d_3/MaxPool:output:01sequential/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?*
paddingVALID*
strides
?
*sequential/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/conv2d_6/BiasAddBiasAdd#sequential/conv2d_6/Conv2D:output:02sequential/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1??
sequential/re_lu_6/ReluRelu$sequential/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????1?}
IdentityIdentity%sequential/re_lu_6/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????1??
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp+^sequential/conv2d_3/BiasAdd/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp*^sequential/conv2d_4/Conv2D/ReadVariableOp*^sequential/conv2d_5/Conv2D/ReadVariableOp+^sequential/conv2d_6/BiasAdd/ReadVariableOp*^sequential/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 2?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12?
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
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
)sequential/conv2d_6/Conv2D/ReadVariableOp)sequential/conv2d_6/Conv2D/ReadVariableOp:S O
0
_output_shapes
:?????????@?

_user_specified_nameX
?
`
D__inference_re_lu_layer_call_and_return_conditional_losses_345049194

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????@?@c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????@?@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?@:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
??
?
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345048813
xJ
0sequential_conv2d_conv2d_readvariableop_resource:@?
1sequential_conv2d_biasadd_readvariableop_resource:@M
2sequential_conv2d_1_conv2d_readvariableop_resource:@?B
3sequential_conv2d_1_biasadd_readvariableop_resource:	?N
2sequential_conv2d_2_conv2d_readvariableop_resource:??B
3sequential_conv2d_2_biasadd_readvariableop_resource:	?N
2sequential_conv2d_3_conv2d_readvariableop_resource:??B
3sequential_conv2d_3_biasadd_readvariableop_resource:	?N
2sequential_conv2d_4_conv2d_readvariableop_resource:??E
6sequential_batch_normalization_readvariableop_resource:	?G
8sequential_batch_normalization_readvariableop_1_resource:	?V
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:	?X
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	?N
2sequential_conv2d_5_conv2d_readvariableop_resource:??G
8sequential_batch_normalization_1_readvariableop_resource:	?I
:sequential_batch_normalization_1_readvariableop_1_resource:	?X
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	?Z
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	?N
2sequential_conv2d_6_conv2d_readvariableop_resource:??B
3sequential_conv2d_6_biasadd_readvariableop_resource:	?
identity??-sequential/batch_normalization/AssignNewValue?/sequential/batch_normalization/AssignNewValue_1?>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?/sequential/batch_normalization_1/AssignNewValue?1sequential/batch_normalization_1/AssignNewValue_1?@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?/sequential/batch_normalization_1/ReadVariableOp?1sequential/batch_normalization_1/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?*sequential/conv2d_3/BiasAdd/ReadVariableOp?)sequential/conv2d_3/Conv2D/ReadVariableOp?)sequential/conv2d_4/Conv2D/ReadVariableOp?)sequential/conv2d_5/Conv2D/ReadVariableOp?*sequential/conv2d_6/BiasAdd/ReadVariableOp?)sequential/conv2d_6/Conv2D/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@|
sequential/re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?@?
 sequential/max_pooling2d/MaxPoolMaxPool#sequential/re_lu/Relu:activations:0*/
_output_shapes
:????????? d@*
ksize
*
paddingVALID*
strides
?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?*
paddingSAME*
strides
?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d??
sequential/re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:????????? d??
"sequential/max_pooling2d_1/MaxPoolMaxPool%sequential/re_lu_1/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2??
sequential/re_lu_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential/conv2d_3/Conv2DConv2D%sequential/re_lu_2/Relu:activations:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
*sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/conv2d_3/BiasAddBiasAdd#sequential/conv2d_3/Conv2D:output:02sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2??
sequential/re_lu_3/ReluRelu$sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
"sequential/max_pooling2d_2/MaxPoolMaxPool%sequential/re_lu_3/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
)sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential/conv2d_4/Conv2DConv2D+sequential/max_pooling2d_2/MaxPool:output:01sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3#sequential/conv2d_4/Conv2D:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
sequential/re_lu_4/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
)sequential/conv2d_5/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential/conv2d_5/Conv2DConv2D%sequential/re_lu_4/Relu:activations:01sequential/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#sequential/conv2d_5/Conv2D:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
/sequential/batch_normalization_1/AssignNewValueAssignVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource>sequential/batch_normalization_1/FusedBatchNormV3:batch_mean:0A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
1sequential/batch_normalization_1/AssignNewValue_1AssignVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceBsequential/batch_normalization_1/FusedBatchNormV3:batch_variance:0C^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
sequential/re_lu_5/ReluRelu5sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
"sequential/max_pooling2d_3/MaxPoolMaxPool%sequential/re_lu_5/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
)sequential/conv2d_6/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential/conv2d_6/Conv2DConv2D+sequential/max_pooling2d_3/MaxPool:output:01sequential/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?*
paddingVALID*
strides
?
*sequential/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/conv2d_6/BiasAddBiasAdd#sequential/conv2d_6/Conv2D:output:02sequential/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1??
sequential/re_lu_6/ReluRelu$sequential/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????1?}
IdentityIdentity%sequential/re_lu_6/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????1??	
NoOpNoOp.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_1?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_10^sequential/batch_normalization_1/AssignNewValue2^sequential/batch_normalization_1/AssignNewValue_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp+^sequential/conv2d_3/BiasAdd/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp*^sequential/conv2d_4/Conv2D/ReadVariableOp*^sequential/conv2d_5/Conv2D/ReadVariableOp+^sequential/conv2d_6/BiasAdd/ReadVariableOp*^sequential/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12b
/sequential/batch_normalization_1/AssignNewValue/sequential/batch_normalization_1/AssignNewValue2f
1sequential/batch_normalization_1/AssignNewValue_11sequential/batch_normalization_1/AssignNewValue_12?
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
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
)sequential/conv2d_6/Conv2D/ReadVariableOp)sequential/conv2d_6/Conv2D/ReadVariableOp:S O
0
_output_shapes
:?????????@?

_user_specified_nameX
?
?
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047763
input_1.
sequential_345047721:@"
sequential_345047723:@/
sequential_345047725:@?#
sequential_345047727:	?0
sequential_345047729:??#
sequential_345047731:	?0
sequential_345047733:??#
sequential_345047735:	?0
sequential_345047737:??#
sequential_345047739:	?#
sequential_345047741:	?#
sequential_345047743:	?#
sequential_345047745:	?0
sequential_345047747:??#
sequential_345047749:	?#
sequential_345047751:	?#
sequential_345047753:	?#
sequential_345047755:	?0
sequential_345047757:??#
sequential_345047759:	?
identity??"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_345047721sequential_345047723sequential_345047725sequential_345047727sequential_345047729sequential_345047731sequential_345047733sequential_345047735sequential_345047737sequential_345047739sequential_345047741sequential_345047743sequential_345047745sequential_345047747sequential_345047749sequential_345047751sequential_345047753sequential_345047755sequential_345047757sequential_345047759* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_345047185?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?k
NoOpNoOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?
G
+__inference_re_lu_5_layer_call_fn_345049478

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_345046896i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?

?
G__inference_conv2d_2_layer_call_and_return_conditional_losses_345046807

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????2?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????2?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
O
3__inference_max_pooling2d_2_layer_call_fn_345049306

inputs
identity?
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
GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_345046599?
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
?
?
,__inference_conv2d_2_layer_call_fn_345049252

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_345046807x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????2?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????2?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_layer_call_fn_345049199

inputs
identity?
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
GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_345046575?
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
?
?
)__inference_model_layer_call_fn_345048183
x
text!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?&

unknown_12:??

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:
identity??StatefulPartitionedCall?
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
:?????????1*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_345048036s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????@?:?????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:?????????@?

_user_specified_nameX:MI
'
_output_shapes
:?????????

_user_specified_nametext
??
?
D__inference_model_layer_call_and_return_conditional_losses_345048513
x
texta
Gvgg__feature_extractor_sequential_conv2d_conv2d_readvariableop_resource:@V
Hvgg__feature_extractor_sequential_conv2d_biasadd_readvariableop_resource:@d
Ivgg__feature_extractor_sequential_conv2d_1_conv2d_readvariableop_resource:@?Y
Jvgg__feature_extractor_sequential_conv2d_1_biasadd_readvariableop_resource:	?e
Ivgg__feature_extractor_sequential_conv2d_2_conv2d_readvariableop_resource:??Y
Jvgg__feature_extractor_sequential_conv2d_2_biasadd_readvariableop_resource:	?e
Ivgg__feature_extractor_sequential_conv2d_3_conv2d_readvariableop_resource:??Y
Jvgg__feature_extractor_sequential_conv2d_3_biasadd_readvariableop_resource:	?e
Ivgg__feature_extractor_sequential_conv2d_4_conv2d_readvariableop_resource:??\
Mvgg__feature_extractor_sequential_batch_normalization_readvariableop_resource:	?^
Ovgg__feature_extractor_sequential_batch_normalization_readvariableop_1_resource:	?m
^vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:	?o
`vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	?e
Ivgg__feature_extractor_sequential_conv2d_5_conv2d_readvariableop_resource:??^
Ovgg__feature_extractor_sequential_batch_normalization_1_readvariableop_resource:	?`
Qvgg__feature_extractor_sequential_batch_normalization_1_readvariableop_1_resource:	?o
`vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	?q
bvgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	?e
Ivgg__feature_extractor_sequential_conv2d_6_conv2d_readvariableop_resource:??Y
Jvgg__feature_extractor_sequential_conv2d_6_biasadd_readvariableop_resource:	?:
'dense_tensordot_readvariableop_resource:	?3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?Dvgg__feature_extractor/sequential/batch_normalization/AssignNewValue?Fvgg__feature_extractor/sequential/batch_normalization/AssignNewValue_1?Uvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?Dvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp?Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1?Fvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue?Hvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue_1?Wvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?Fvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp?Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1??vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp?>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp?Avgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp?@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp?Avgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp?@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp?Avgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp?@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp?@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp?@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp?Avgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp?@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp?
>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpGvgg__feature_extractor_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
/vgg__feature_extractor/sequential/conv2d/Conv2DConv2DxFvgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
?
?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpHvgg__feature_extractor_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
0vgg__feature_extractor/sequential/conv2d/BiasAddBiasAdd8vgg__feature_extractor/sequential/conv2d/Conv2D:output:0Gvgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@?
,vgg__feature_extractor/sequential/re_lu/ReluRelu9vgg__feature_extractor/sequential/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?@?
7vgg__feature_extractor/sequential/max_pooling2d/MaxPoolMaxPool:vgg__feature_extractor/sequential/re_lu/Relu:activations:0*/
_output_shapes
:????????? d@*
ksize
*
paddingVALID*
strides
?
@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
1vgg__feature_extractor/sequential/conv2d_1/Conv2DConv2D@vgg__feature_extractor/sequential/max_pooling2d/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?*
paddingSAME*
strides
?
Avgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2vgg__feature_extractor/sequential/conv2d_1/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_1/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d??
.vgg__feature_extractor/sequential/re_lu_1/ReluRelu;vgg__feature_extractor/sequential/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:????????? d??
9vgg__feature_extractor/sequential/max_pooling2d_1/MaxPoolMaxPool<vgg__feature_extractor/sequential/re_lu_1/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
1vgg__feature_extractor/sequential/conv2d_2/Conv2DConv2DBvgg__feature_extractor/sequential/max_pooling2d_1/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
Avgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2vgg__feature_extractor/sequential/conv2d_2/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_2/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2??
.vgg__feature_extractor/sequential/re_lu_2/ReluRelu;vgg__feature_extractor/sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
1vgg__feature_extractor/sequential/conv2d_3/Conv2DConv2D<vgg__feature_extractor/sequential/re_lu_2/Relu:activations:0Hvgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
Avgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2vgg__feature_extractor/sequential/conv2d_3/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_3/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2??
.vgg__feature_extractor/sequential/re_lu_3/ReluRelu;vgg__feature_extractor/sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
9vgg__feature_extractor/sequential/max_pooling2d_2/MaxPoolMaxPool<vgg__feature_extractor/sequential/re_lu_3/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
1vgg__feature_extractor/sequential/conv2d_4/Conv2DConv2DBvgg__feature_extractor/sequential/max_pooling2d_2/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
Dvgg__feature_extractor/sequential/batch_normalization/ReadVariableOpReadVariableOpMvgg__feature_extractor_sequential_batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1ReadVariableOpOvgg__feature_extractor_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Uvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp^vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Fvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3:vgg__feature_extractor/sequential/conv2d_4/Conv2D:output:0Lvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp:value:0Nvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1:value:0]vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0_vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
Dvgg__feature_extractor/sequential/batch_normalization/AssignNewValueAssignVariableOp^vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceSvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3:batch_mean:0V^vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
Fvgg__feature_extractor/sequential/batch_normalization/AssignNewValue_1AssignVariableOp`vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceWvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3:batch_variance:0X^vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
.vgg__feature_extractor/sequential/re_lu_4/ReluReluJvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
1vgg__feature_extractor/sequential/conv2d_5/Conv2DConv2D<vgg__feature_extractor/sequential/re_lu_4/Relu:activations:0Hvgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
Fvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpReadVariableOpOvgg__feature_extractor_sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOpQvgg__feature_extractor_sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Wvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp`vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbvgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Hvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3:vgg__feature_extractor/sequential/conv2d_5/Conv2D:output:0Nvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp:value:0Pvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1:value:0_vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0avgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
Fvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValueAssignVariableOp`vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceUvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3:batch_mean:0X^vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
Hvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue_1AssignVariableOpbvgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceYvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3:batch_variance:0Z^vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
.vgg__feature_extractor/sequential/re_lu_5/ReluReluLvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
9vgg__feature_extractor/sequential/max_pooling2d_3/MaxPoolMaxPool<vgg__feature_extractor/sequential/re_lu_5/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
1vgg__feature_extractor/sequential/conv2d_6/Conv2DConv2DBvgg__feature_extractor/sequential/max_pooling2d_3/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?*
paddingVALID*
strides
?
Avgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2vgg__feature_extractor/sequential/conv2d_6/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_6/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1??
.vgg__feature_extractor/sequential/re_lu_6/ReluRelu;vgg__feature_extractor/sequential/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????1?g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
	transpose	Transpose<vgg__feature_extractor/sequential/re_lu_6/Relu:activations:0transpose/perm:output:0*
T0*0
_output_shapes
:?????????1?l
*adaptive_average_pooling2d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 adaptive_average_pooling2d/splitSplit3adaptive_average_pooling2d/split/split_dim:output:0transpose:y:0*
T0*?

_output_shapes?

?
:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*
	num_split1?
 adaptive_average_pooling2d/stackPack)adaptive_average_pooling2d/split:output:0)adaptive_average_pooling2d/split:output:1)adaptive_average_pooling2d/split:output:2)adaptive_average_pooling2d/split:output:3)adaptive_average_pooling2d/split:output:4)adaptive_average_pooling2d/split:output:5)adaptive_average_pooling2d/split:output:6)adaptive_average_pooling2d/split:output:7)adaptive_average_pooling2d/split:output:8)adaptive_average_pooling2d/split:output:9*adaptive_average_pooling2d/split:output:10*adaptive_average_pooling2d/split:output:11*adaptive_average_pooling2d/split:output:12*adaptive_average_pooling2d/split:output:13*adaptive_average_pooling2d/split:output:14*adaptive_average_pooling2d/split:output:15*adaptive_average_pooling2d/split:output:16*adaptive_average_pooling2d/split:output:17*adaptive_average_pooling2d/split:output:18*adaptive_average_pooling2d/split:output:19*adaptive_average_pooling2d/split:output:20*adaptive_average_pooling2d/split:output:21*adaptive_average_pooling2d/split:output:22*adaptive_average_pooling2d/split:output:23*adaptive_average_pooling2d/split:output:24*adaptive_average_pooling2d/split:output:25*adaptive_average_pooling2d/split:output:26*adaptive_average_pooling2d/split:output:27*adaptive_average_pooling2d/split:output:28*adaptive_average_pooling2d/split:output:29*adaptive_average_pooling2d/split:output:30*adaptive_average_pooling2d/split:output:31*adaptive_average_pooling2d/split:output:32*adaptive_average_pooling2d/split:output:33*adaptive_average_pooling2d/split:output:34*adaptive_average_pooling2d/split:output:35*adaptive_average_pooling2d/split:output:36*adaptive_average_pooling2d/split:output:37*adaptive_average_pooling2d/split:output:38*adaptive_average_pooling2d/split:output:39*adaptive_average_pooling2d/split:output:40*adaptive_average_pooling2d/split:output:41*adaptive_average_pooling2d/split:output:42*adaptive_average_pooling2d/split:output:43*adaptive_average_pooling2d/split:output:44*adaptive_average_pooling2d/split:output:45*adaptive_average_pooling2d/split:output:46*adaptive_average_pooling2d/split:output:47*adaptive_average_pooling2d/split:output:48*
N1*
T0*4
_output_shapes"
 :?????????1?*

axisn
,adaptive_average_pooling2d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
"adaptive_average_pooling2d/split_1Split5adaptive_average_pooling2d/split_1/split_dim:output:0)adaptive_average_pooling2d/stack:output:0*
T0*4
_output_shapes"
 :?????????1?*
	num_split?
"adaptive_average_pooling2d/stack_1Pack+adaptive_average_pooling2d/split_1:output:0*
N*
T0*8
_output_shapes&
$:"?????????1?*

axis?
1adaptive_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
adaptive_average_pooling2d/MeanMean+adaptive_average_pooling2d/stack_1:output:0:adaptive_average_pooling2d/Mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????1??
SqueezeSqueeze(adaptive_average_pooling2d/Mean:output:0*
T0*,
_output_shapes
:?????????1?*
squeeze_dims
?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	?*
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
value	B : ?
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
value	B : ?
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
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	TransposeSqueeze:output:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????1??
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????1~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????1i
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????1?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOpE^vgg__feature_extractor/sequential/batch_normalization/AssignNewValueG^vgg__feature_extractor/sequential/batch_normalization/AssignNewValue_1V^vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpX^vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1E^vgg__feature_extractor/sequential/batch_normalization/ReadVariableOpG^vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1G^vgg__feature_extractor/sequential/batch_normalization_1/AssignNewValueI^vgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue_1X^vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpZ^vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1G^vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpI^vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1@^vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp?^vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????@?:?????????: : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2?
Dvgg__feature_extractor/sequential/batch_normalization/AssignNewValueDvgg__feature_extractor/sequential/batch_normalization/AssignNewValue2?
Fvgg__feature_extractor/sequential/batch_normalization/AssignNewValue_1Fvgg__feature_extractor/sequential/batch_normalization/AssignNewValue_12?
Uvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpUvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12?
Dvgg__feature_extractor/sequential/batch_normalization/ReadVariableOpDvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp2?
Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_12?
Fvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValueFvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue2?
Hvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue_1Hvgg__feature_extractor/sequential/batch_normalization_1/AssignNewValue_12?
Wvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpWvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12?
Fvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpFvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp2?
Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_12?
?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp2?
>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp2?
Avgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp2?
@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp2?
Avgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp2?
@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp2?
Avgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp2?
@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp2?
@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp2?
@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp2?
Avgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp2?
@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp:S O
0
_output_shapes
:?????????@?

_user_specified_nameX:MI
'
_output_shapes
:?????????

_user_specified_nametext
?
?
D__inference_dense_layer_call_and_return_conditional_losses_345048917

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
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
value	B : ?
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
value	B : ?
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
value	B : ?
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
:?????????1??
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????1?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:?????????1?
 
_user_specified_nameinputs
?
?
:__inference_vgg__feature_extractor_layer_call_fn_345048655
x!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?&

unknown_12:??

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?
identity??StatefulPartitionedCall?
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
:?????????1?*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047585x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:?????????@?

_user_specified_nameX
?
?
)__inference_dense_layer_call_fn_345048887

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_345047908s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????1?: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????1?
 
_user_specified_nameinputs
?

?
G__inference_conv2d_2_layer_call_and_return_conditional_losses_345049262

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????2?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????2?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
E
)__inference_re_lu_layer_call_fn_345049189

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_345046770i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????@?@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?@:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
G
+__inference_re_lu_4_layer_call_fn_345049392

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_345046869i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
?
G__inference_conv2d_5_layer_call_and_return_conditional_losses_345046878

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????2?^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????2?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?

?
E__inference_conv2d_layer_call_and_return_conditional_losses_345049184

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????@?@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
?
.__inference_sequential_layer_call_fn_345048962

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?&

unknown_12:??

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?
identity??StatefulPartitionedCall?
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
:?????????1?*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_345046923x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?

?
G__inference_conv2d_3_layer_call_and_return_conditional_losses_345049291

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????2?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????2?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
??
?
D__inference_model_layer_call_and_return_conditional_losses_345048348
x
texta
Gvgg__feature_extractor_sequential_conv2d_conv2d_readvariableop_resource:@V
Hvgg__feature_extractor_sequential_conv2d_biasadd_readvariableop_resource:@d
Ivgg__feature_extractor_sequential_conv2d_1_conv2d_readvariableop_resource:@?Y
Jvgg__feature_extractor_sequential_conv2d_1_biasadd_readvariableop_resource:	?e
Ivgg__feature_extractor_sequential_conv2d_2_conv2d_readvariableop_resource:??Y
Jvgg__feature_extractor_sequential_conv2d_2_biasadd_readvariableop_resource:	?e
Ivgg__feature_extractor_sequential_conv2d_3_conv2d_readvariableop_resource:??Y
Jvgg__feature_extractor_sequential_conv2d_3_biasadd_readvariableop_resource:	?e
Ivgg__feature_extractor_sequential_conv2d_4_conv2d_readvariableop_resource:??\
Mvgg__feature_extractor_sequential_batch_normalization_readvariableop_resource:	?^
Ovgg__feature_extractor_sequential_batch_normalization_readvariableop_1_resource:	?m
^vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:	?o
`vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	?e
Ivgg__feature_extractor_sequential_conv2d_5_conv2d_readvariableop_resource:??^
Ovgg__feature_extractor_sequential_batch_normalization_1_readvariableop_resource:	?`
Qvgg__feature_extractor_sequential_batch_normalization_1_readvariableop_1_resource:	?o
`vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	?q
bvgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	?e
Ivgg__feature_extractor_sequential_conv2d_6_conv2d_readvariableop_resource:??Y
Jvgg__feature_extractor_sequential_conv2d_6_biasadd_readvariableop_resource:	?:
'dense_tensordot_readvariableop_resource:	?3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?Uvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?Dvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp?Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1?Wvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?Fvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp?Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1??vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp?>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp?Avgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp?@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp?Avgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp?@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp?Avgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp?@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp?@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp?@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp?Avgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp?@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp?
>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpGvgg__feature_extractor_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
/vgg__feature_extractor/sequential/conv2d/Conv2DConv2DxFvgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
?
?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpHvgg__feature_extractor_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
0vgg__feature_extractor/sequential/conv2d/BiasAddBiasAdd8vgg__feature_extractor/sequential/conv2d/Conv2D:output:0Gvgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@?
,vgg__feature_extractor/sequential/re_lu/ReluRelu9vgg__feature_extractor/sequential/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?@?
7vgg__feature_extractor/sequential/max_pooling2d/MaxPoolMaxPool:vgg__feature_extractor/sequential/re_lu/Relu:activations:0*/
_output_shapes
:????????? d@*
ksize
*
paddingVALID*
strides
?
@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
1vgg__feature_extractor/sequential/conv2d_1/Conv2DConv2D@vgg__feature_extractor/sequential/max_pooling2d/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?*
paddingSAME*
strides
?
Avgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2vgg__feature_extractor/sequential/conv2d_1/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_1/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d??
.vgg__feature_extractor/sequential/re_lu_1/ReluRelu;vgg__feature_extractor/sequential/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:????????? d??
9vgg__feature_extractor/sequential/max_pooling2d_1/MaxPoolMaxPool<vgg__feature_extractor/sequential/re_lu_1/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
1vgg__feature_extractor/sequential/conv2d_2/Conv2DConv2DBvgg__feature_extractor/sequential/max_pooling2d_1/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
Avgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2vgg__feature_extractor/sequential/conv2d_2/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_2/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2??
.vgg__feature_extractor/sequential/re_lu_2/ReluRelu;vgg__feature_extractor/sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
1vgg__feature_extractor/sequential/conv2d_3/Conv2DConv2D<vgg__feature_extractor/sequential/re_lu_2/Relu:activations:0Hvgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
Avgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2vgg__feature_extractor/sequential/conv2d_3/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_3/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2??
.vgg__feature_extractor/sequential/re_lu_3/ReluRelu;vgg__feature_extractor/sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
9vgg__feature_extractor/sequential/max_pooling2d_2/MaxPoolMaxPool<vgg__feature_extractor/sequential/re_lu_3/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
1vgg__feature_extractor/sequential/conv2d_4/Conv2DConv2DBvgg__feature_extractor/sequential/max_pooling2d_2/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
Dvgg__feature_extractor/sequential/batch_normalization/ReadVariableOpReadVariableOpMvgg__feature_extractor_sequential_batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1ReadVariableOpOvgg__feature_extractor_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Uvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp^vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Fvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3:vgg__feature_extractor/sequential/conv2d_4/Conv2D:output:0Lvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp:value:0Nvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1:value:0]vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0_vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
is_training( ?
.vgg__feature_extractor/sequential/re_lu_4/ReluReluJvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
1vgg__feature_extractor/sequential/conv2d_5/Conv2DConv2D<vgg__feature_extractor/sequential/re_lu_4/Relu:activations:0Hvgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
Fvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpReadVariableOpOvgg__feature_extractor_sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOpQvgg__feature_extractor_sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Wvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp`vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbvgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Hvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3:vgg__feature_extractor/sequential/conv2d_5/Conv2D:output:0Nvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp:value:0Pvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1:value:0_vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0avgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
is_training( ?
.vgg__feature_extractor/sequential/re_lu_5/ReluReluLvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
9vgg__feature_extractor/sequential/max_pooling2d_3/MaxPoolMaxPool<vgg__feature_extractor/sequential/re_lu_5/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOpReadVariableOpIvgg__feature_extractor_sequential_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
1vgg__feature_extractor/sequential/conv2d_6/Conv2DConv2DBvgg__feature_extractor/sequential/max_pooling2d_3/MaxPool:output:0Hvgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?*
paddingVALID*
strides
?
Avgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpJvgg__feature_extractor_sequential_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2vgg__feature_extractor/sequential/conv2d_6/BiasAddBiasAdd:vgg__feature_extractor/sequential/conv2d_6/Conv2D:output:0Ivgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1??
.vgg__feature_extractor/sequential/re_lu_6/ReluRelu;vgg__feature_extractor/sequential/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????1?g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
	transpose	Transpose<vgg__feature_extractor/sequential/re_lu_6/Relu:activations:0transpose/perm:output:0*
T0*0
_output_shapes
:?????????1?l
*adaptive_average_pooling2d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 adaptive_average_pooling2d/splitSplit3adaptive_average_pooling2d/split/split_dim:output:0transpose:y:0*
T0*?

_output_shapes?

?
:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*
	num_split1?
 adaptive_average_pooling2d/stackPack)adaptive_average_pooling2d/split:output:0)adaptive_average_pooling2d/split:output:1)adaptive_average_pooling2d/split:output:2)adaptive_average_pooling2d/split:output:3)adaptive_average_pooling2d/split:output:4)adaptive_average_pooling2d/split:output:5)adaptive_average_pooling2d/split:output:6)adaptive_average_pooling2d/split:output:7)adaptive_average_pooling2d/split:output:8)adaptive_average_pooling2d/split:output:9*adaptive_average_pooling2d/split:output:10*adaptive_average_pooling2d/split:output:11*adaptive_average_pooling2d/split:output:12*adaptive_average_pooling2d/split:output:13*adaptive_average_pooling2d/split:output:14*adaptive_average_pooling2d/split:output:15*adaptive_average_pooling2d/split:output:16*adaptive_average_pooling2d/split:output:17*adaptive_average_pooling2d/split:output:18*adaptive_average_pooling2d/split:output:19*adaptive_average_pooling2d/split:output:20*adaptive_average_pooling2d/split:output:21*adaptive_average_pooling2d/split:output:22*adaptive_average_pooling2d/split:output:23*adaptive_average_pooling2d/split:output:24*adaptive_average_pooling2d/split:output:25*adaptive_average_pooling2d/split:output:26*adaptive_average_pooling2d/split:output:27*adaptive_average_pooling2d/split:output:28*adaptive_average_pooling2d/split:output:29*adaptive_average_pooling2d/split:output:30*adaptive_average_pooling2d/split:output:31*adaptive_average_pooling2d/split:output:32*adaptive_average_pooling2d/split:output:33*adaptive_average_pooling2d/split:output:34*adaptive_average_pooling2d/split:output:35*adaptive_average_pooling2d/split:output:36*adaptive_average_pooling2d/split:output:37*adaptive_average_pooling2d/split:output:38*adaptive_average_pooling2d/split:output:39*adaptive_average_pooling2d/split:output:40*adaptive_average_pooling2d/split:output:41*adaptive_average_pooling2d/split:output:42*adaptive_average_pooling2d/split:output:43*adaptive_average_pooling2d/split:output:44*adaptive_average_pooling2d/split:output:45*adaptive_average_pooling2d/split:output:46*adaptive_average_pooling2d/split:output:47*adaptive_average_pooling2d/split:output:48*
N1*
T0*4
_output_shapes"
 :?????????1?*

axisn
,adaptive_average_pooling2d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
"adaptive_average_pooling2d/split_1Split5adaptive_average_pooling2d/split_1/split_dim:output:0)adaptive_average_pooling2d/stack:output:0*
T0*4
_output_shapes"
 :?????????1?*
	num_split?
"adaptive_average_pooling2d/stack_1Pack+adaptive_average_pooling2d/split_1:output:0*
N*
T0*8
_output_shapes&
$:"?????????1?*

axis?
1adaptive_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
adaptive_average_pooling2d/MeanMean+adaptive_average_pooling2d/stack_1:output:0:adaptive_average_pooling2d/Mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????1??
SqueezeSqueeze(adaptive_average_pooling2d/Mean:output:0*
T0*,
_output_shapes
:?????????1?*
squeeze_dims
?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	?*
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
value	B : ?
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
value	B : ?
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
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	TransposeSqueeze:output:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????1??
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????1~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????1i
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????1?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOpV^vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpX^vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1E^vgg__feature_extractor/sequential/batch_normalization/ReadVariableOpG^vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1X^vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpZ^vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1G^vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpI^vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1@^vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp?^vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpB^vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpA^vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????@?:?????????: : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2?
Uvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpUvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Wvgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12?
Dvgg__feature_extractor/sequential/batch_normalization/ReadVariableOpDvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp2?
Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1Fvgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_12?
Wvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpWvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Yvgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12?
Fvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpFvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp2?
Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1Hvgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_12?
?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp?vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp2?
>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp>vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp2?
Avgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp2?
@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp2?
Avgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp2?
@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp2?
Avgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp2?
@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp2?
@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp2?
@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp2?
Avgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpAvgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp2?
@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp@vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp:S O
0
_output_shapes
:?????????@?

_user_specified_nameX:MI
'
_output_shapes
:?????????

_user_specified_nametext
?

?
G__inference_conv2d_6_layer_call_and_return_conditional_losses_345049512

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????1?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????2?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_345049243

inputs
identity?
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
?
?
)__inference_model_layer_call_fn_345048133
x
text!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?&

unknown_12:??

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:
identity??StatefulPartitionedCall?
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
:?????????1*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_345047915s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????@?:?????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:?????????@?

_user_specified_nameX:MI
'
_output_shapes
:?????????

_user_specified_nametext
?
?
.__inference_sequential_layer_call_fn_345047273
conv2d_input!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?&

unknown_12:??

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?
identity??StatefulPartitionedCall?
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
:?????????1?*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_345047185x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:?????????@?
&
_user_specified_nameconv2d_input
?T
?

I__inference_sequential_layer_call_and_return_conditional_losses_345047337
conv2d_input*
conv2d_345047276:@
conv2d_345047278:@-
conv2d_1_345047283:@?!
conv2d_1_345047285:	?.
conv2d_2_345047290:??!
conv2d_2_345047292:	?.
conv2d_3_345047296:??!
conv2d_3_345047298:	?.
conv2d_4_345047303:??,
batch_normalization_345047306:	?,
batch_normalization_345047308:	?,
batch_normalization_345047310:	?,
batch_normalization_345047312:	?.
conv2d_5_345047316:??.
batch_normalization_1_345047319:	?.
batch_normalization_1_345047321:	?.
batch_normalization_1_345047323:	?.
batch_normalization_1_345047325:	?.
conv2d_6_345047330:??!
conv2d_6_345047332:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_345047276conv2d_345047278*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_345046759?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_345046770?
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
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
GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_345046575?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_345047283conv2d_1_345047285*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? d?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_345046783?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? d?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_345046794?
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_345046587?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_345047290conv2d_2_345047292*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_345046807?
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_345046818?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_345047296conv2d_3_345047298*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_345046830?
re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_345046841?
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_345046599?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_4_345047303*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_345046851?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_345047306batch_normalization_345047308batch_normalization_345047310batch_normalization_345047312*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345046624?
re_lu_4/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_345046869?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_5_345047316*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_345046878?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_1_345047319batch_normalization_1_345047321batch_normalization_1_345047323batch_normalization_1_345047325*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345046688?
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_345046896?
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_345046739?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_6_345047330conv2d_6_345047332*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_345046909?
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_345046920x
IdentityIdentity re_lu_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1??
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:^ Z
0
_output_shapes
:?????????@?
&
_user_specified_nameconv2d_input
?"
?

D__inference_model_layer_call_and_return_conditional_losses_345047915
x
text:
 vgg__feature_extractor_345047772:@.
 vgg__feature_extractor_345047774:@;
 vgg__feature_extractor_345047776:@?/
 vgg__feature_extractor_345047778:	?<
 vgg__feature_extractor_345047780:??/
 vgg__feature_extractor_345047782:	?<
 vgg__feature_extractor_345047784:??/
 vgg__feature_extractor_345047786:	?<
 vgg__feature_extractor_345047788:??/
 vgg__feature_extractor_345047790:	?/
 vgg__feature_extractor_345047792:	?/
 vgg__feature_extractor_345047794:	?/
 vgg__feature_extractor_345047796:	?<
 vgg__feature_extractor_345047798:??/
 vgg__feature_extractor_345047800:	?/
 vgg__feature_extractor_345047802:	?/
 vgg__feature_extractor_345047804:	?/
 vgg__feature_extractor_345047806:	?<
 vgg__feature_extractor_345047808:??/
 vgg__feature_extractor_345047810:	?"
dense_345047909:	?
dense_345047911:
identity??dense/StatefulPartitionedCall?.vgg__feature_extractor/StatefulPartitionedCall?
.vgg__feature_extractor/StatefulPartitionedCallStatefulPartitionedCallx vgg__feature_extractor_345047772 vgg__feature_extractor_345047774 vgg__feature_extractor_345047776 vgg__feature_extractor_345047778 vgg__feature_extractor_345047780 vgg__feature_extractor_345047782 vgg__feature_extractor_345047784 vgg__feature_extractor_345047786 vgg__feature_extractor_345047788 vgg__feature_extractor_345047790 vgg__feature_extractor_345047792 vgg__feature_extractor_345047794 vgg__feature_extractor_345047796 vgg__feature_extractor_345047798 vgg__feature_extractor_345047800 vgg__feature_extractor_345047802 vgg__feature_extractor_345047804 vgg__feature_extractor_345047806 vgg__feature_extractor_345047808 vgg__feature_extractor_345047810* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047450g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
	transpose	Transpose7vgg__feature_extractor/StatefulPartitionedCall:output:0transpose/perm:output:0*
T0*0
_output_shapes
:?????????1??
*adaptive_average_pooling2d/PartitionedCallPartitionedCalltranspose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *b
f]R[
Y__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_345047875?
SqueezeSqueeze3adaptive_average_pooling2d/PartitionedCall:output:0*
T0*,
_output_shapes
:?????????1?*
squeeze_dims
?
dense/StatefulPartitionedCallStatefulPartitionedCallSqueeze:output:0dense_345047909dense_345047911*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_345047908y
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????1?
NoOpNoOp^dense/StatefulPartitionedCall/^vgg__feature_extractor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????@?:?????????: : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.vgg__feature_extractor/StatefulPartitionedCall.vgg__feature_extractor/StatefulPartitionedCall:S O
0
_output_shapes
:?????????@?

_user_specified_nameX:MI
'
_output_shapes
:?????????

_user_specified_nametext
?
?
,__inference_conv2d_3_layer_call_fn_345049281

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_345046830x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????2?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????2?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
b
F__inference_re_lu_2_layer_call_and_return_conditional_losses_345049272

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????2?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
?
,__inference_conv2d_4_layer_call_fn_345049318

inputs#
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_345046851x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????2?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????2?: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?T
?

I__inference_sequential_layer_call_and_return_conditional_losses_345047185

inputs*
conv2d_345047124:@
conv2d_345047126:@-
conv2d_1_345047131:@?!
conv2d_1_345047133:	?.
conv2d_2_345047138:??!
conv2d_2_345047140:	?.
conv2d_3_345047144:??!
conv2d_3_345047146:	?.
conv2d_4_345047151:??,
batch_normalization_345047154:	?,
batch_normalization_345047156:	?,
batch_normalization_345047158:	?,
batch_normalization_345047160:	?.
conv2d_5_345047164:??.
batch_normalization_1_345047167:	?.
batch_normalization_1_345047169:	?.
batch_normalization_1_345047171:	?.
batch_normalization_1_345047173:	?.
conv2d_6_345047178:??!
conv2d_6_345047180:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_345047124conv2d_345047126*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_345046759?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_345046770?
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
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
GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_345046575?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_345047131conv2d_1_345047133*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? d?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_345046783?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? d?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_345046794?
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_345046587?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_345047138conv2d_2_345047140*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_345046807?
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_345046818?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_345047144conv2d_3_345047146*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_345046830?
re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_345046841?
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_345046599?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_4_345047151*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_345046851?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_345047154batch_normalization_345047156batch_normalization_345047158batch_normalization_345047160*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345046655?
re_lu_4/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_345046869?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_5_345047164*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_345046878?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_1_345047167batch_normalization_1_345047169batch_normalization_1_345047171batch_normalization_1_345047173*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345046719?
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_345046896?
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_345046739?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_6_345047178conv2d_6_345047180*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_345046909?
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_345046920x
IdentityIdentity re_lu_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1??
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:X T
0
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?

?
G__inference_conv2d_6_layer_call_and_return_conditional_losses_345046909

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????1?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????2?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?

?
G__inference_conv2d_1_layer_call_and_return_conditional_losses_345046783

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:????????? d?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? d@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? d@
 
_user_specified_nameinputs
?
?
:__inference_vgg__feature_extractor_layer_call_fn_345047673
input_1!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?&

unknown_12:??

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?
identity??StatefulPartitionedCall?
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
:?????????1?*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047585x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?
?
:__inference_vgg__feature_extractor_layer_call_fn_345047493
input_1!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?&

unknown_12:??

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?
identity??StatefulPartitionedCall?
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
:?????????1?*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047450x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?
b
F__inference_re_lu_6_layer_call_and_return_conditional_losses_345046920

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????1?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????1?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????1?:X T
0
_output_shapes
:?????????1?
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345046688

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_re_lu_1_layer_call_and_return_conditional_losses_345049233

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:????????? d?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:????????? d?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:????????? d?:X T
0
_output_shapes
:????????? d?
 
_user_specified_nameinputs
?
G
+__inference_re_lu_6_layer_call_fn_345049517

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_345046920i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????1?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????1?:X T
0
_output_shapes
:?????????1?
 
_user_specified_nameinputs
?
?
G__inference_conv2d_5_layer_call_and_return_conditional_losses_345049411

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????2?^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????2?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
b
F__inference_re_lu_3_layer_call_and_return_conditional_losses_345049301

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????2?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?2
?	
"__inference__traced_save_345049612
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

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B,Prediction/kernel/.ATTRIBUTES/VARIABLE_VALUEB*Prediction/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_model_dense_kernel_read_readvariableop+savev2_model_dense_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?::@:@:@?:?:??:?:??:?:??:?:?:?:?:??:?:?:?:?:??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.	*
(
_output_shapes
:??:!


_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:

_output_shapes
: 
?
?
.__inference_sequential_layer_call_fn_345046966
conv2d_input!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?&

unknown_12:??

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?
identity??StatefulPartitionedCall?
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
:?????????1?*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_345046923x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:?????????@?
&
_user_specified_nameconv2d_input
?
j
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_345046587

inputs
identity?
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
?
?
*__inference_conv2d_layer_call_fn_345049174

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_345046759x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@?@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_layer_call_fn_345049351

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345046655?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_layer_call_fn_345049007

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?&

unknown_12:??

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?
identity??StatefulPartitionedCall?
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
:?????????1?*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_345047185x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345046624

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_re_lu_2_layer_call_fn_345049267

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_345046818i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345046719

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_5_layer_call_fn_345049404

inputs#
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_345046878x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????2?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????2?: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
Z
>__inference_adaptive_average_pooling2d_layer_call_fn_345048818

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *b
f]R[
Y__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_345047875i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????1?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????1?:X T
0
_output_shapes
:?????????1?
 
_user_specified_nameinputs
?
?
,__inference_conv2d_6_layer_call_fn_345049502

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????1?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_345046909x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????1?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????2?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?	
?
9__inference_batch_normalization_1_layer_call_fn_345049437

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345046719?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_re_lu_6_layer_call_and_return_conditional_losses_345049522

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????1?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????1?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????1?:X T
0
_output_shapes
:?????????1?
 
_user_specified_nameinputs
?n
?
I__inference_sequential_layer_call_and_return_conditional_losses_345049165

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@B
'conv2d_1_conv2d_readvariableop_resource:@?7
(conv2d_1_biasadd_readvariableop_resource:	?C
'conv2d_2_conv2d_readvariableop_resource:??7
(conv2d_2_biasadd_readvariableop_resource:	?C
'conv2d_3_conv2d_readvariableop_resource:??7
(conv2d_3_biasadd_readvariableop_resource:	?C
'conv2d_4_conv2d_readvariableop_resource:??:
+batch_normalization_readvariableop_resource:	?<
-batch_normalization_readvariableop_1_resource:	?K
<batch_normalization_fusedbatchnormv3_readvariableop_resource:	?M
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	?C
'conv2d_5_conv2d_readvariableop_resource:??<
-batch_normalization_1_readvariableop_resource:	?>
/batch_normalization_1_readvariableop_1_resource:	?M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	?C
'conv2d_6_conv2d_readvariableop_resource:??7
(conv2d_6_biasadd_readvariableop_resource:	?
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@f

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?@?
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:????????? d@*
ksize
*
paddingVALID*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?j
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:????????? d??
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?j
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_3/Conv2DConv2Dre_lu_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?j
re_lu_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
max_pooling2d_2/MaxPoolMaxPoolre_lu_3/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0y
re_lu_4/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_5/Conv2DConv2Dre_lu_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0{
re_lu_5/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
max_pooling2d_3/MaxPoolMaxPoolre_lu_5/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_6/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?*
paddingVALID*
strides
?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?j
re_lu_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????1?r
IdentityIdentityre_lu_6/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????1??
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????@?: : : : : : : : : : : : : : : : : : : : 2H
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
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_345049311

inputs
identity?
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
?
u
Y__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_345048878

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0inputs*
T0*?

_output_shapes?

?
:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*
	num_split1?
stackPacksplit:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7split:output:8split:output:9split:output:10split:output:11split:output:12split:output:13split:output:14split:output:15split:output:16split:output:17split:output:18split:output:19split:output:20split:output:21split:output:22split:output:23split:output:24split:output:25split:output:26split:output:27split:output:28split:output:29split:output:30split:output:31split:output:32split:output:33split:output:34split:output:35split:output:36split:output:37split:output:38split:output:39split:output:40split:output:41split:output:42split:output:43split:output:44split:output:45split:output:46split:output:47split:output:48*
N1*
T0*4
_output_shapes"
 :?????????1?*

axisS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0stack:output:0*
T0*4
_output_shapes"
 :?????????1?*
	num_splity
stack_1Packsplit_1:output:0*
N*
T0*8
_output_shapes&
$:"?????????1?*

axisg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      z
MeanMeanstack_1:output:0Mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????1?^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:?????????1?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????1?:X T
0
_output_shapes
:?????????1?
 
_user_specified_nameinputs
?
G
+__inference_re_lu_3_layer_call_fn_345049296

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????2?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_345046841i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345049387

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
9__inference_batch_normalization_1_layer_call_fn_345049424

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345046688?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345049473

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
O
3__inference_max_pooling2d_3_layer_call_fn_345049488

inputs
identity?
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
GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_345046739?
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
?
h
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_345046575

inputs
identity?
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
?
b
F__inference_re_lu_4_layer_call_and_return_conditional_losses_345049397

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????2?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345049369

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
$__inference__wrapped_model_345046566

args_0

args_1g
Mmodel_vgg__feature_extractor_sequential_conv2d_conv2d_readvariableop_resource:@\
Nmodel_vgg__feature_extractor_sequential_conv2d_biasadd_readvariableop_resource:@j
Omodel_vgg__feature_extractor_sequential_conv2d_1_conv2d_readvariableop_resource:@?_
Pmodel_vgg__feature_extractor_sequential_conv2d_1_biasadd_readvariableop_resource:	?k
Omodel_vgg__feature_extractor_sequential_conv2d_2_conv2d_readvariableop_resource:??_
Pmodel_vgg__feature_extractor_sequential_conv2d_2_biasadd_readvariableop_resource:	?k
Omodel_vgg__feature_extractor_sequential_conv2d_3_conv2d_readvariableop_resource:??_
Pmodel_vgg__feature_extractor_sequential_conv2d_3_biasadd_readvariableop_resource:	?k
Omodel_vgg__feature_extractor_sequential_conv2d_4_conv2d_readvariableop_resource:??b
Smodel_vgg__feature_extractor_sequential_batch_normalization_readvariableop_resource:	?d
Umodel_vgg__feature_extractor_sequential_batch_normalization_readvariableop_1_resource:	?s
dmodel_vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:	?u
fmodel_vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	?k
Omodel_vgg__feature_extractor_sequential_conv2d_5_conv2d_readvariableop_resource:??d
Umodel_vgg__feature_extractor_sequential_batch_normalization_1_readvariableop_resource:	?f
Wmodel_vgg__feature_extractor_sequential_batch_normalization_1_readvariableop_1_resource:	?u
fmodel_vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	?w
hmodel_vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	?k
Omodel_vgg__feature_extractor_sequential_conv2d_6_conv2d_readvariableop_resource:??_
Pmodel_vgg__feature_extractor_sequential_conv2d_6_biasadd_readvariableop_resource:	?@
-model_dense_tensordot_readvariableop_resource:	?9
+model_dense_biasadd_readvariableop_resource:
identity??"model/dense/BiasAdd/ReadVariableOp?$model/dense/Tensordot/ReadVariableOp?[model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?]model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?Jmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp?Lmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1?]model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?_model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?Lmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp?Nmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1?Emodel/vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp?Dmodel/vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp?Gmodel/vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp?Fmodel/vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp?Gmodel/vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp?Fmodel/vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp?Gmodel/vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp?Fmodel/vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp?Fmodel/vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp?Fmodel/vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp?Gmodel/vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp?Fmodel/vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp?
Dmodel/vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpMmodel_vgg__feature_extractor_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
5model/vgg__feature_extractor/sequential/conv2d/Conv2DConv2Dargs_0Lmodel/vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
?
Emodel/vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpNmodel_vgg__feature_extractor_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
6model/vgg__feature_extractor/sequential/conv2d/BiasAddBiasAdd>model/vgg__feature_extractor/sequential/conv2d/Conv2D:output:0Mmodel/vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@?
2model/vgg__feature_extractor/sequential/re_lu/ReluRelu?model/vgg__feature_extractor/sequential/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?@?
=model/vgg__feature_extractor/sequential/max_pooling2d/MaxPoolMaxPool@model/vgg__feature_extractor/sequential/re_lu/Relu:activations:0*/
_output_shapes
:????????? d@*
ksize
*
paddingVALID*
strides
?
Fmodel/vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOpOmodel_vgg__feature_extractor_sequential_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
7model/vgg__feature_extractor/sequential/conv2d_1/Conv2DConv2DFmodel/vgg__feature_extractor/sequential/max_pooling2d/MaxPool:output:0Nmodel/vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d?*
paddingSAME*
strides
?
Gmodel/vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpPmodel_vgg__feature_extractor_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8model/vgg__feature_extractor/sequential/conv2d_1/BiasAddBiasAdd@model/vgg__feature_extractor/sequential/conv2d_1/Conv2D:output:0Omodel/vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? d??
4model/vgg__feature_extractor/sequential/re_lu_1/ReluReluAmodel/vgg__feature_extractor/sequential/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:????????? d??
?model/vgg__feature_extractor/sequential/max_pooling2d_1/MaxPoolMaxPoolBmodel/vgg__feature_extractor/sequential/re_lu_1/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
Fmodel/vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOpOmodel_vgg__feature_extractor_sequential_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
7model/vgg__feature_extractor/sequential/conv2d_2/Conv2DConv2DHmodel/vgg__feature_extractor/sequential/max_pooling2d_1/MaxPool:output:0Nmodel/vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
Gmodel/vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpPmodel_vgg__feature_extractor_sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8model/vgg__feature_extractor/sequential/conv2d_2/BiasAddBiasAdd@model/vgg__feature_extractor/sequential/conv2d_2/Conv2D:output:0Omodel/vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2??
4model/vgg__feature_extractor/sequential/re_lu_2/ReluReluAmodel/vgg__feature_extractor/sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
Fmodel/vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOpOmodel_vgg__feature_extractor_sequential_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
7model/vgg__feature_extractor/sequential/conv2d_3/Conv2DConv2DBmodel/vgg__feature_extractor/sequential/re_lu_2/Relu:activations:0Nmodel/vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
Gmodel/vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpPmodel_vgg__feature_extractor_sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8model/vgg__feature_extractor/sequential/conv2d_3/BiasAddBiasAdd@model/vgg__feature_extractor/sequential/conv2d_3/Conv2D:output:0Omodel/vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2??
4model/vgg__feature_extractor/sequential/re_lu_3/ReluReluAmodel/vgg__feature_extractor/sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2??
?model/vgg__feature_extractor/sequential/max_pooling2d_2/MaxPoolMaxPoolBmodel/vgg__feature_extractor/sequential/re_lu_3/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
Fmodel/vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOpOmodel_vgg__feature_extractor_sequential_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
7model/vgg__feature_extractor/sequential/conv2d_4/Conv2DConv2DHmodel/vgg__feature_extractor/sequential/max_pooling2d_2/MaxPool:output:0Nmodel/vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
Jmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOpReadVariableOpSmodel_vgg__feature_extractor_sequential_batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Lmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1ReadVariableOpUmodel_vgg__feature_extractor_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
[model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpdmodel_vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
]model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpfmodel_vgg__feature_extractor_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Lmodel/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3@model/vgg__feature_extractor/sequential/conv2d_4/Conv2D:output:0Rmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp:value:0Tmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1:value:0cmodel/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0emodel/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
is_training( ?
4model/vgg__feature_extractor/sequential/re_lu_4/ReluReluPmodel/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
Fmodel/vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpReadVariableOpOmodel_vgg__feature_extractor_sequential_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
7model/vgg__feature_extractor/sequential/conv2d_5/Conv2DConv2DBmodel/vgg__feature_extractor/sequential/re_lu_4/Relu:activations:0Nmodel/vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2?*
paddingSAME*
strides
?
Lmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpReadVariableOpUmodel_vgg__feature_extractor_sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Nmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOpWmodel_vgg__feature_extractor_sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
]model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpfmodel_vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
_model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOphmodel_vgg__feature_extractor_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Nmodel/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3@model/vgg__feature_extractor/sequential/conv2d_5/Conv2D:output:0Tmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp:value:0Vmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1:value:0emodel/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0gmodel/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????2?:?:?:?:?:*
epsilon%o?:*
is_training( ?
4model/vgg__feature_extractor/sequential/re_lu_5/ReluReluRmodel/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2??
?model/vgg__feature_extractor/sequential/max_pooling2d_3/MaxPoolMaxPoolBmodel/vgg__feature_extractor/sequential/re_lu_5/Relu:activations:0*0
_output_shapes
:?????????2?*
ksize
*
paddingVALID*
strides
?
Fmodel/vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOpReadVariableOpOmodel_vgg__feature_extractor_sequential_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
7model/vgg__feature_extractor/sequential/conv2d_6/Conv2DConv2DHmodel/vgg__feature_extractor/sequential/max_pooling2d_3/MaxPool:output:0Nmodel/vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1?*
paddingVALID*
strides
?
Gmodel/vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpPmodel_vgg__feature_extractor_sequential_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8model/vgg__feature_extractor/sequential/conv2d_6/BiasAddBiasAdd@model/vgg__feature_extractor/sequential/conv2d_6/Conv2D:output:0Omodel/vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????1??
4model/vgg__feature_extractor/sequential/re_lu_6/ReluReluAmodel/vgg__feature_extractor/sequential/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????1?m
model/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
model/transpose	TransposeBmodel/vgg__feature_extractor/sequential/re_lu_6/Relu:activations:0model/transpose/perm:output:0*
T0*0
_output_shapes
:?????????1?r
0model/adaptive_average_pooling2d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&model/adaptive_average_pooling2d/splitSplit9model/adaptive_average_pooling2d/split/split_dim:output:0model/transpose:y:0*
T0*?

_output_shapes?

?
:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*
	num_split1?
&model/adaptive_average_pooling2d/stackPack/model/adaptive_average_pooling2d/split:output:0/model/adaptive_average_pooling2d/split:output:1/model/adaptive_average_pooling2d/split:output:2/model/adaptive_average_pooling2d/split:output:3/model/adaptive_average_pooling2d/split:output:4/model/adaptive_average_pooling2d/split:output:5/model/adaptive_average_pooling2d/split:output:6/model/adaptive_average_pooling2d/split:output:7/model/adaptive_average_pooling2d/split:output:8/model/adaptive_average_pooling2d/split:output:90model/adaptive_average_pooling2d/split:output:100model/adaptive_average_pooling2d/split:output:110model/adaptive_average_pooling2d/split:output:120model/adaptive_average_pooling2d/split:output:130model/adaptive_average_pooling2d/split:output:140model/adaptive_average_pooling2d/split:output:150model/adaptive_average_pooling2d/split:output:160model/adaptive_average_pooling2d/split:output:170model/adaptive_average_pooling2d/split:output:180model/adaptive_average_pooling2d/split:output:190model/adaptive_average_pooling2d/split:output:200model/adaptive_average_pooling2d/split:output:210model/adaptive_average_pooling2d/split:output:220model/adaptive_average_pooling2d/split:output:230model/adaptive_average_pooling2d/split:output:240model/adaptive_average_pooling2d/split:output:250model/adaptive_average_pooling2d/split:output:260model/adaptive_average_pooling2d/split:output:270model/adaptive_average_pooling2d/split:output:280model/adaptive_average_pooling2d/split:output:290model/adaptive_average_pooling2d/split:output:300model/adaptive_average_pooling2d/split:output:310model/adaptive_average_pooling2d/split:output:320model/adaptive_average_pooling2d/split:output:330model/adaptive_average_pooling2d/split:output:340model/adaptive_average_pooling2d/split:output:350model/adaptive_average_pooling2d/split:output:360model/adaptive_average_pooling2d/split:output:370model/adaptive_average_pooling2d/split:output:380model/adaptive_average_pooling2d/split:output:390model/adaptive_average_pooling2d/split:output:400model/adaptive_average_pooling2d/split:output:410model/adaptive_average_pooling2d/split:output:420model/adaptive_average_pooling2d/split:output:430model/adaptive_average_pooling2d/split:output:440model/adaptive_average_pooling2d/split:output:450model/adaptive_average_pooling2d/split:output:460model/adaptive_average_pooling2d/split:output:470model/adaptive_average_pooling2d/split:output:48*
N1*
T0*4
_output_shapes"
 :?????????1?*

axist
2model/adaptive_average_pooling2d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
(model/adaptive_average_pooling2d/split_1Split;model/adaptive_average_pooling2d/split_1/split_dim:output:0/model/adaptive_average_pooling2d/stack:output:0*
T0*4
_output_shapes"
 :?????????1?*
	num_split?
(model/adaptive_average_pooling2d/stack_1Pack1model/adaptive_average_pooling2d/split_1:output:0*
N*
T0*8
_output_shapes&
$:"?????????1?*

axis?
7model/adaptive_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
%model/adaptive_average_pooling2d/MeanMean1model/adaptive_average_pooling2d/stack_1:output:0@model/adaptive_average_pooling2d/Mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????1??
model/SqueezeSqueeze.model/adaptive_average_pooling2d/Mean:output:0*
T0*,
_output_shapes
:?????????1?*
squeeze_dims
?
$model/dense/Tensordot/ReadVariableOpReadVariableOp-model_dense_tensordot_readvariableop_resource*
_output_shapes
:	?*
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
value	B : ?
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
value	B : ?
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
valueB: ?
model/dense/Tensordot/ProdProd'model/dense/Tensordot/GatherV2:output:0$model/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: g
model/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
model/dense/Tensordot/Prod_1Prod)model/dense/Tensordot/GatherV2_1:output:0&model/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!model/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
model/dense/Tensordot/concatConcatV2#model/dense/Tensordot/free:output:0#model/dense/Tensordot/axes:output:0*model/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
model/dense/Tensordot/stackPack#model/dense/Tensordot/Prod:output:0%model/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
model/dense/Tensordot/transpose	Transposemodel/Squeeze:output:0%model/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????1??
model/dense/Tensordot/ReshapeReshape#model/dense/Tensordot/transpose:y:0$model/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
model/dense/Tensordot/MatMulMatMul&model/dense/Tensordot/Reshape:output:0,model/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
model/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:e
#model/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
model/dense/Tensordot/concat_1ConcatV2'model/dense/Tensordot/GatherV2:output:0&model/dense/Tensordot/Const_2:output:0,model/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
model/dense/TensordotReshape&model/dense/Tensordot/MatMul:product:0'model/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????1?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/dense/BiasAddBiasAddmodel/dense/Tensordot:output:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????1o
IdentityIdentitymodel/dense/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????1?
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/Tensordot/ReadVariableOp\^model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp^^model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1K^model/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOpM^model/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1^^model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp`^model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1M^model/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpO^model/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1F^model/vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOpE^model/vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpH^model/vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpG^model/vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpH^model/vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpG^model/vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpH^model/vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpG^model/vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpG^model/vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpG^model/vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpH^model/vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpG^model/vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????@?:?????????: : : : : : : : : : : : : : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/Tensordot/ReadVariableOp$model/dense/Tensordot/ReadVariableOp2?
[model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp[model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
]model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1]model/vgg__feature_extractor/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12?
Jmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOpJmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp2?
Lmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_1Lmodel/vgg__feature_extractor/sequential/batch_normalization/ReadVariableOp_12?
]model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp]model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
_model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1_model/vgg__feature_extractor/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12?
Lmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOpLmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp2?
Nmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_1Nmodel/vgg__feature_extractor/sequential/batch_normalization_1/ReadVariableOp_12?
Emodel/vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOpEmodel/vgg__feature_extractor/sequential/conv2d/BiasAdd/ReadVariableOp2?
Dmodel/vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOpDmodel/vgg__feature_extractor/sequential/conv2d/Conv2D/ReadVariableOp2?
Gmodel/vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOpGmodel/vgg__feature_extractor/sequential/conv2d_1/BiasAdd/ReadVariableOp2?
Fmodel/vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOpFmodel/vgg__feature_extractor/sequential/conv2d_1/Conv2D/ReadVariableOp2?
Gmodel/vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOpGmodel/vgg__feature_extractor/sequential/conv2d_2/BiasAdd/ReadVariableOp2?
Fmodel/vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOpFmodel/vgg__feature_extractor/sequential/conv2d_2/Conv2D/ReadVariableOp2?
Gmodel/vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOpGmodel/vgg__feature_extractor/sequential/conv2d_3/BiasAdd/ReadVariableOp2?
Fmodel/vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOpFmodel/vgg__feature_extractor/sequential/conv2d_3/Conv2D/ReadVariableOp2?
Fmodel/vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOpFmodel/vgg__feature_extractor/sequential/conv2d_4/Conv2D/ReadVariableOp2?
Fmodel/vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOpFmodel/vgg__feature_extractor/sequential/conv2d_5/Conv2D/ReadVariableOp2?
Gmodel/vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOpGmodel/vgg__feature_extractor/sequential/conv2d_6/BiasAdd/ReadVariableOp2?
Fmodel/vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOpFmodel/vgg__feature_extractor/sequential/conv2d_6/Conv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@?
 
_user_specified_nameargs_0:OK
'
_output_shapes
:?????????
 
_user_specified_nameargs_1
?
b
F__inference_re_lu_3_layer_call_and_return_conditional_losses_345046841

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????2?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?:X T
0
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
G
+__inference_re_lu_1_layer_call_fn_345049228

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? d?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_345046794i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:????????? d?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:????????? d?:X T
0
_output_shapes
:????????? d?
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
B
args_08
serving_default_args_0:0?????????@?
9
args_1/
serving_default_args_1:0?????????@
output_14
StatefulPartitionedCall:0?????????1tensorflow/serving/predict:??
?
FeatureExtraction
AdaptiveAvgPool
flatten

Prediction
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_model
?
output_channel
ConvNet
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_model
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719
20
21"
trackable_list_wrapper
?
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
111
212
313
614
715
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_model_layer_call_fn_345048133
)__inference_model_layer_call_fn_345048183?
???
FullArgSpec,
args$?!
jself
jX
jtext

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_model_layer_call_and_return_conditional_losses_345048348
D__inference_model_layer_call_and_return_conditional_losses_345048513?
???
FullArgSpec,
args$?!
jself
jX
jtext

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
$__inference__wrapped_model_345046566args_0args_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
=serving_default"
signature_map
 "
trackable_list_wrapper
?
>layer_with_weights-0
>layer-0
?layer-1
@layer-2
Alayer_with_weights-1
Alayer-3
Blayer-4
Clayer-5
Dlayer_with_weights-2
Dlayer-6
Elayer-7
Flayer_with_weights-3
Flayer-8
Glayer-9
Hlayer-10
Ilayer_with_weights-4
Ilayer-11
Jlayer_with_weights-5
Jlayer-12
Klayer-13
Llayer_with_weights-6
Llayer-14
Mlayer_with_weights-7
Mlayer-15
Nlayer-16
Olayer-17
Player_with_weights-8
Player-18
Qlayer-19
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719"
trackable_list_wrapper
?
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
111
212
313
614
715"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
:__inference_vgg__feature_extractor_layer_call_fn_345047493
:__inference_vgg__feature_extractor_layer_call_fn_345048610
:__inference_vgg__feature_extractor_layer_call_fn_345048655
:__inference_vgg__feature_extractor_layer_call_fn_345047673?
???
FullArgSpec$
args?
jself
jX

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345048734
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345048813
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047718
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047763?
???
FullArgSpec$
args?
jself
jX

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
>__inference_adaptive_average_pooling2d_layer_call_fn_345048818?
???
FullArgSpec
args?
jself
jinputs
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Y__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_345048878?
???
FullArgSpec
args?
jself
jinputs
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
%:#	?2model/dense/kernel
:2model/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_layer_call_fn_345048887?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_layer_call_and_return_conditional_losses_345048917?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
':%@2conv2d/kernel
:@2conv2d/bias
*:(@?2conv2d_1/kernel
:?2conv2d_1/bias
+:)??2conv2d_2/kernel
:?2conv2d_2/bias
+:)??2conv2d_3/kernel
:?2conv2d_3/bias
+:)??2conv2d_4/kernel
(:&?2batch_normalization/gamma
':%?2batch_normalization/beta
0:.? (2batch_normalization/moving_mean
4:2? (2#batch_normalization/moving_variance
+:)??2conv2d_5/kernel
*:(?2batch_normalization_1/gamma
):'?2batch_normalization_1/beta
2:0? (2!batch_normalization_1/moving_mean
6:4? (2%batch_normalization_1/moving_variance
+:)??2conv2d_6/kernel
:?2conv2d_6/bias
<
/0
01
42
53"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_signature_wrapper_345048565args_0args_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?

$kernel
%bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
?
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
?
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
?

&kernel
'bias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

(kernel
)bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
+bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

,kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	-gamma
.beta
/moving_mean
0moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

1kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	2gamma
3beta
4moving_mean
5moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

6kernel
7bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719"
trackable_list_wrapper
?
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
111
212
313
614
715"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_sequential_layer_call_fn_345046966
.__inference_sequential_layer_call_fn_345048962
.__inference_sequential_layer_call_fn_345049007
.__inference_sequential_layer_call_fn_345047273?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_layer_call_and_return_conditional_losses_345049086
I__inference_sequential_layer_call_and_return_conditional_losses_345049165
I__inference_sequential_layer_call_and_return_conditional_losses_345047337
I__inference_sequential_layer_call_and_return_conditional_losses_345047401?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
<
/0
01
42
53"
trackable_list_wrapper
'
0"
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
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_layer_call_fn_345049174?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_layer_call_and_return_conditional_losses_345049184?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_re_lu_layer_call_fn_345049189?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_re_lu_layer_call_and_return_conditional_losses_345049194?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
?2?
1__inference_max_pooling2d_layer_call_fn_345049199?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_345049204?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_conv2d_1_layer_call_fn_345049213?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_1_layer_call_and_return_conditional_losses_345049223?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_re_lu_1_layer_call_fn_345049228?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_1_layer_call_and_return_conditional_losses_345049233?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
3__inference_max_pooling2d_1_layer_call_fn_345049238?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_345049243?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_conv2d_2_layer_call_fn_345049252?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_2_layer_call_and_return_conditional_losses_345049262?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_re_lu_2_layer_call_fn_345049267?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_2_layer_call_and_return_conditional_losses_345049272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_conv2d_3_layer_call_fn_345049281?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_3_layer_call_and_return_conditional_losses_345049291?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_re_lu_3_layer_call_fn_345049296?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_3_layer_call_and_return_conditional_losses_345049301?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
3__inference_max_pooling2d_2_layer_call_fn_345049306?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_345049311?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
,0"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_conv2d_4_layer_call_fn_345049318?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_4_layer_call_and_return_conditional_losses_345049325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
-0
.1
/2
03"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
7__inference_batch_normalization_layer_call_fn_345049338
7__inference_batch_normalization_layer_call_fn_345049351?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345049369
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345049387?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_re_lu_4_layer_call_fn_345049392?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_4_layer_call_and_return_conditional_losses_345049397?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
10"
trackable_list_wrapper
'
10"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_conv2d_5_layer_call_fn_345049404?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_5_layer_call_and_return_conditional_losses_345049411?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
20
31
42
53"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
9__inference_batch_normalization_1_layer_call_fn_345049424
9__inference_batch_normalization_1_layer_call_fn_345049437?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345049455
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345049473?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_re_lu_5_layer_call_fn_345049478?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_5_layer_call_and_return_conditional_losses_345049483?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
3__inference_max_pooling2d_3_layer_call_fn_345049488?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_345049493?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_conv2d_6_layer_call_fn_345049502?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_6_layer_call_and_return_conditional_losses_345049512?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_re_lu_6_layer_call_fn_345049517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_re_lu_6_layer_call_and_return_conditional_losses_345049522?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
<
/0
01
42
53"
trackable_list_wrapper
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19"
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
/0
01"
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
40
51"
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
trackable_dict_wrapper?
$__inference__wrapped_model_345046566?$%&'()*+,-./01234567Z?W
P?M
)?&
args_0?????????@?
 ?
args_1?????????
? "7?4
2
output_1&?#
output_1?????????1?
Y__inference_adaptive_average_pooling2d_layer_call_and_return_conditional_losses_345048878j8?5
.?+
)?&
inputs?????????1?
? ".?+
$?!
0?????????1?
? ?
>__inference_adaptive_average_pooling2d_layer_call_fn_345048818]8?5
.?+
)?&
inputs?????????1?
? "!??????????1??
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345049455?2345N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345049473?2345N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
9__inference_batch_normalization_1_layer_call_fn_345049424?2345N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
9__inference_batch_normalization_1_layer_call_fn_345049437?2345N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345049369?-./0N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_345049387?-./0N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
7__inference_batch_normalization_layer_call_fn_345049338?-./0N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
7__inference_batch_normalization_layer_call_fn_345049351?-./0N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
G__inference_conv2d_1_layer_call_and_return_conditional_losses_345049223m&'7?4
-?*
(?%
inputs????????? d@
? ".?+
$?!
0????????? d?
? ?
,__inference_conv2d_1_layer_call_fn_345049213`&'7?4
-?*
(?%
inputs????????? d@
? "!?????????? d??
G__inference_conv2d_2_layer_call_and_return_conditional_losses_345049262n()8?5
.?+
)?&
inputs?????????2?
? ".?+
$?!
0?????????2?
? ?
,__inference_conv2d_2_layer_call_fn_345049252a()8?5
.?+
)?&
inputs?????????2?
? "!??????????2??
G__inference_conv2d_3_layer_call_and_return_conditional_losses_345049291n*+8?5
.?+
)?&
inputs?????????2?
? ".?+
$?!
0?????????2?
? ?
,__inference_conv2d_3_layer_call_fn_345049281a*+8?5
.?+
)?&
inputs?????????2?
? "!??????????2??
G__inference_conv2d_4_layer_call_and_return_conditional_losses_345049325m,8?5
.?+
)?&
inputs?????????2?
? ".?+
$?!
0?????????2?
? ?
,__inference_conv2d_4_layer_call_fn_345049318`,8?5
.?+
)?&
inputs?????????2?
? "!??????????2??
G__inference_conv2d_5_layer_call_and_return_conditional_losses_345049411m18?5
.?+
)?&
inputs?????????2?
? ".?+
$?!
0?????????2?
? ?
,__inference_conv2d_5_layer_call_fn_345049404`18?5
.?+
)?&
inputs?????????2?
? "!??????????2??
G__inference_conv2d_6_layer_call_and_return_conditional_losses_345049512n678?5
.?+
)?&
inputs?????????2?
? ".?+
$?!
0?????????1?
? ?
,__inference_conv2d_6_layer_call_fn_345049502a678?5
.?+
)?&
inputs?????????2?
? "!??????????1??
E__inference_conv2d_layer_call_and_return_conditional_losses_345049184n$%8?5
.?+
)?&
inputs?????????@?
? ".?+
$?!
0?????????@?@
? ?
*__inference_conv2d_layer_call_fn_345049174a$%8?5
.?+
)?&
inputs?????????@?
? "!??????????@?@?
D__inference_dense_layer_call_and_return_conditional_losses_345048917e4?1
*?'
%?"
inputs?????????1?
? ")?&
?
0?????????1
? ?
)__inference_dense_layer_call_fn_345048887X4?1
*?'
%?"
inputs?????????1?
? "??????????1?
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_345049243?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
3__inference_max_pooling2d_1_layer_call_fn_345049238?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_345049311?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
3__inference_max_pooling2d_2_layer_call_fn_345049306?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_345049493?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
3__inference_max_pooling2d_3_layer_call_fn_345049488?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_345049204?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_layer_call_fn_345049199?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
D__inference_model_layer_call_and_return_conditional_losses_345048348?$%&'()*+,-./01234567W?T
M?J
$?!
X?????????@?
?
text?????????
p 
? ")?&
?
0?????????1
? ?
D__inference_model_layer_call_and_return_conditional_losses_345048513?$%&'()*+,-./01234567W?T
M?J
$?!
X?????????@?
?
text?????????
p
? ")?&
?
0?????????1
? ?
)__inference_model_layer_call_fn_345048133?$%&'()*+,-./01234567W?T
M?J
$?!
X?????????@?
?
text?????????
p 
? "??????????1?
)__inference_model_layer_call_fn_345048183?$%&'()*+,-./01234567W?T
M?J
$?!
X?????????@?
?
text?????????
p
? "??????????1?
F__inference_re_lu_1_layer_call_and_return_conditional_losses_345049233j8?5
.?+
)?&
inputs????????? d?
? ".?+
$?!
0????????? d?
? ?
+__inference_re_lu_1_layer_call_fn_345049228]8?5
.?+
)?&
inputs????????? d?
? "!?????????? d??
F__inference_re_lu_2_layer_call_and_return_conditional_losses_345049272j8?5
.?+
)?&
inputs?????????2?
? ".?+
$?!
0?????????2?
? ?
+__inference_re_lu_2_layer_call_fn_345049267]8?5
.?+
)?&
inputs?????????2?
? "!??????????2??
F__inference_re_lu_3_layer_call_and_return_conditional_losses_345049301j8?5
.?+
)?&
inputs?????????2?
? ".?+
$?!
0?????????2?
? ?
+__inference_re_lu_3_layer_call_fn_345049296]8?5
.?+
)?&
inputs?????????2?
? "!??????????2??
F__inference_re_lu_4_layer_call_and_return_conditional_losses_345049397j8?5
.?+
)?&
inputs?????????2?
? ".?+
$?!
0?????????2?
? ?
+__inference_re_lu_4_layer_call_fn_345049392]8?5
.?+
)?&
inputs?????????2?
? "!??????????2??
F__inference_re_lu_5_layer_call_and_return_conditional_losses_345049483j8?5
.?+
)?&
inputs?????????2?
? ".?+
$?!
0?????????2?
? ?
+__inference_re_lu_5_layer_call_fn_345049478]8?5
.?+
)?&
inputs?????????2?
? "!??????????2??
F__inference_re_lu_6_layer_call_and_return_conditional_losses_345049522j8?5
.?+
)?&
inputs?????????1?
? ".?+
$?!
0?????????1?
? ?
+__inference_re_lu_6_layer_call_fn_345049517]8?5
.?+
)?&
inputs?????????1?
? "!??????????1??
D__inference_re_lu_layer_call_and_return_conditional_losses_345049194j8?5
.?+
)?&
inputs?????????@?@
? ".?+
$?!
0?????????@?@
? ?
)__inference_re_lu_layer_call_fn_345049189]8?5
.?+
)?&
inputs?????????@?@
? "!??????????@?@?
I__inference_sequential_layer_call_and_return_conditional_losses_345047337?$%&'()*+,-./01234567F?C
<?9
/?,
conv2d_input?????????@?
p 

 
? ".?+
$?!
0?????????1?
? ?
I__inference_sequential_layer_call_and_return_conditional_losses_345047401?$%&'()*+,-./01234567F?C
<?9
/?,
conv2d_input?????????@?
p

 
? ".?+
$?!
0?????????1?
? ?
I__inference_sequential_layer_call_and_return_conditional_losses_345049086?$%&'()*+,-./01234567@?=
6?3
)?&
inputs?????????@?
p 

 
? ".?+
$?!
0?????????1?
? ?
I__inference_sequential_layer_call_and_return_conditional_losses_345049165?$%&'()*+,-./01234567@?=
6?3
)?&
inputs?????????@?
p

 
? ".?+
$?!
0?????????1?
? ?
.__inference_sequential_layer_call_fn_345046966?$%&'()*+,-./01234567F?C
<?9
/?,
conv2d_input?????????@?
p 

 
? "!??????????1??
.__inference_sequential_layer_call_fn_345047273?$%&'()*+,-./01234567F?C
<?9
/?,
conv2d_input?????????@?
p

 
? "!??????????1??
.__inference_sequential_layer_call_fn_345048962{$%&'()*+,-./01234567@?=
6?3
)?&
inputs?????????@?
p 

 
? "!??????????1??
.__inference_sequential_layer_call_fn_345049007{$%&'()*+,-./01234567@?=
6?3
)?&
inputs?????????@?
p

 
? "!??????????1??
'__inference_signature_wrapper_345048565?$%&'()*+,-./01234567n?k
? 
d?a
3
args_0)?&
args_0?????????@?
*
args_1 ?
args_1?????????"7?4
2
output_1&?#
output_1?????????1?
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047718?$%&'()*+,-./01234567=?:
3?0
*?'
input_1?????????@?
p 
? ".?+
$?!
0?????????1?
? ?
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345047763?$%&'()*+,-./01234567=?:
3?0
*?'
input_1?????????@?
p
? ".?+
$?!
0?????????1?
? ?
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345048734$%&'()*+,-./012345677?4
-?*
$?!
X?????????@?
p 
? ".?+
$?!
0?????????1?
? ?
U__inference_vgg__feature_extractor_layer_call_and_return_conditional_losses_345048813$%&'()*+,-./012345677?4
-?*
$?!
X?????????@?
p
? ".?+
$?!
0?????????1?
? ?
:__inference_vgg__feature_extractor_layer_call_fn_345047493x$%&'()*+,-./01234567=?:
3?0
*?'
input_1?????????@?
p 
? "!??????????1??
:__inference_vgg__feature_extractor_layer_call_fn_345047673x$%&'()*+,-./01234567=?:
3?0
*?'
input_1?????????@?
p
? "!??????????1??
:__inference_vgg__feature_extractor_layer_call_fn_345048610r$%&'()*+,-./012345677?4
-?*
$?!
X?????????@?
p 
? "!??????????1??
:__inference_vgg__feature_extractor_layer_call_fn_345048655r$%&'()*+,-./012345677?4
-?*
$?!
X?????????@?
p
? "!??????????1?