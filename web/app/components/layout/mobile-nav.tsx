import { Camera, Clock, Settings } from "lucide-react"
import { NavLink } from "react-router-dom"

import { cn } from "@/lib/utils"

const NAV_ITEMS = [
  { to: "/", label: "Camera", icon: Camera, end: true },
  { to: "/history", label: "History", icon: Clock },
  { to: "/settings", label: "Settings", icon: Settings },
]

export function MobileNav() {
  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-50 flex h-[60px] border-t bg-card md:hidden"
      aria-label="Main navigation"
    >
      {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          className={({ isActive }) =>
            cn(
              "flex flex-1 flex-col items-center justify-center gap-1 text-xs font-medium transition-colors",
              "text-muted-foreground hover:text-secondary-foreground",
              isActive && "text-primary",
            )
          }
        >
          <Icon size={22} aria-hidden="true" />
          <span className="text-[10px] uppercase tracking-wider">{label}</span>
        </NavLink>
      ))}
    </nav>
  )
}
