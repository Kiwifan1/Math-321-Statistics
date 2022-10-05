.global _start
.text

_start:
    movl $132, %r8d ; 132 = 0x84 = 0b10000100
    movl $219, %r9d ; 219 = 0xDB = 0b11011011
    movl %r8d, %r10d ; r10d = 0x84
    subl %r9d, %r10d ; 219 - 132 = 87 = 0xA9 = 0b01010111
    negl %r10d ; -87 = 0xA9 = 0b10101001
    movq $60, %rax
    xor %rdi, %rdi
    syscall