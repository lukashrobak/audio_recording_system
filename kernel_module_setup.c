# Setup for building and installing the Kernel Module for RPi5 gp0_clk 12.288MHz (only v1)
# Bachelors Project — Design and Implementation of a System for Long-Term Audio Recording (Lukas Hrobak)

mkdir ~/gpclk_module && cd ~/gpclk_module

# CREATE THE C FILE:

nano gpclk0.c

#include <linux/module.h>
#include <linux/clk.h>
#include <linux/kprobes.h>

static struct clk *gp0_clk;

static struct kprobe kp = { .symbol_name = "kallsyms_lookup_name" };
typedef unsigned long (*kallsyms_fn)(const char *name);
typedef struct clk *(*clk_lookup_fn)(const char *name);

static int __init gpclk_init(void)
{
    kallsyms_fn kallsyms_lookup_name;
    clk_lookup_fn __clk_lookup;

    if (register_kprobe(&kp) < 0) {
        pr_err("gpclk0: kprobe failed\n");
        return -ENOENT;
    }
    kallsyms_lookup_name = (kallsyms_fn)kp.addr;
    unregister_kprobe(&kp);

    __clk_lookup = (clk_lookup_fn)kallsyms_lookup_name("__clk_lookup");
    if (!__clk_lookup) {
        pr_err("gpclk0: cannot find __clk_lookup\n");
        return -ENOENT;
    }

    gp0_clk = __clk_lookup("clk_gp0");
    if (!gp0_clk) {
        pr_err("gpclk0: __clk_lookup returned NULL\n");
        return -ENOENT;
    }

    clk_set_rate(gp0_clk, 12288000);
    clk_prepare_enable(gp0_clk);
    pr_info("gpclk0: running at %lu Hz\n", clk_get_rate(gp0_clk));
    return 0;
}

static void __exit gpclk_exit(void)
{
    if (gp0_clk) {
        clk_disable_unprepare(gp0_clk);
    }
}

module_init(gpclk_init);
module_exit(gpclk_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Enable clk_gp0 on RPi5");

# CREATE THE MAKEFILE:

nano Makefile

obj-m += gpclk0.o
all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean


# BUILD AND INSTALL:

make
sudo insmod gpclk0.ko
sudo cp gpclk0.ko /lib/modules/$(uname -r)/
sudo depmod -a
echo "gpclk0" | sudo tee /etc/modules-load.d/gpclk0.conf
