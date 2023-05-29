#python3 -m zinc --loader_type=constant --work_type=test --batch_size=512
#python3 -m zinc --loader_type=constant --work_type=test --batch_size=512 --aot

python3 -m zinc --work_type=test --loader_type=gnm 
python3 -m zinc --work_type=test --loader_type=gnm --aot
python3 -m zinc --work_type=test --loader_type=normal
