#python3 -m zinc --loader_type=constant --work_type=train --batch_size=512
#python3 -m zinc --loader_type=constant --work_type=train --batch_size=512 --aot

python3 -m zinc --work_type=train --batch_size=512 --loader_type=gnm 
python3 -m zinc --work_type=train --batch_size=512 --loader_type=gnm --aot
python3 -m zinc --work_type=train --batch_size=512 --loader_type=normal
