import { Outlet, NavLink } from "react-router-dom"
import { Camera, Clock, Settings } from "lucide-react"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
} from "@/components/ui/sidebar"
import { MobileNav } from "./mobile-nav"

const NAV_ITEMS = [
  { to: "/", label: "Camera", icon: Camera, end: true },
  { to: "/history", label: "History", icon: Clock },
  { to: "/settings", label: "Settings", icon: Settings },
]

export function AppShell() {
  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full bg-background transition-colors duration-300">
        {/* Desktop Sidebar - Using exactly the same glass effect as the Settings header */}
        <Sidebar className="border-r border-border transition-colors duration-300">
          <SidebarHeader className="border-b border-border/50 p-4">
            <div className="flex items-center gap-3">
              {/* Premium Logo Container */}
              <div className="flex size-9 items-center justify-center rounded-xl border border-border/50 bg-muted/50 shadow-inner">
                <img src="/pwa-192x192.png" alt="Mina logo" className="size-6 object-contain" />
              </div>
              <span className="font-mono text-lg font-semibold tracking-wide text-foreground">
                Mina
              </span>
            </div>
          </SidebarHeader>

          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupContent>
                <SidebarMenu className="gap-2 px-4 py-6">
                  {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
                    <SidebarMenuItem key={to}>
                      <NavLink to={to} end={end}>
                        {({ isActive }: { isActive: boolean }) => (
                          <SidebarMenuButton
                            isActive={isActive}
                            // Applying the exact same styling logic as your Theme Switcher
                            className={`min-h-11 rounded-xl px-4 transition-all duration-300 ${
                              isActive
                                ? "bg-background text-foreground shadow-sm ring-1 ring-border/50"
                                : "text-muted-foreground hover:bg-muted hover:text-foreground"
                            }`}
                          >
                            <Icon
                              size={18}
                              aria-hidden="true"
                              className={isActive ? "text-foreground" : "text-muted-foreground"}
                            />
                            <span className="font-mono text-[13px] font-medium tracking-wider uppercase">
                              {label}
                            </span>
                          </SidebarMenuButton>
                        )}
                      </NavLink>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>

          <SidebarFooter className="border-t border-border/50 p-5">
            <span className="w-fit shrink-0 rounded-full border border-border/50 bg-muted/50 px-3 py-1 font-mono text-[10px] font-bold tracking-widest text-muted-foreground shadow-inner uppercase">
              v1.0.0
            </span>
          </SidebarFooter>
        </Sidebar>

        {/* Main Content Area */}
        <SidebarInset className="bg-transparent">
          <main className="flex h-full flex-1 flex-col pb-[60px] md:pb-0" id="main-content">
            <Outlet />
          </main>
        </SidebarInset>

        {/* Mobile Bottom Nav */}
        <MobileNav />
      </div>
    </SidebarProvider>
  )
}
