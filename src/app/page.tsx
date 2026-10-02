import Link from "next/link";
import { SIMULATIONS, SUBJECTS } from "@/lib/subjects";

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-blue-500/30 font-sans">
      {/* Hero Section */}
      <section className="relative px-6 py-24 md:py-32 max-w-5xl mx-auto flex flex-col items-center text-center">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/20 via-slate-950 to-slate-950 -z-10"></div>
        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
          EduSim
        </h1>
        <p className="text-xl md:text-2xl text-slate-400 font-light mb-12 max-w-2xl">
          GPU-Powered Interactive Science Simulations
        </p>
        
        {/* Subject Categories */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full">
          {Object.entries(SUBJECTS).map(([key, meta]) => {
            const count = SIMULATIONS.filter(s => s.subject === key).length;
            return (
              <div key={key} className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 flex flex-col items-center gap-3 transition-colors hover:bg-slate-800/50">
                <span className="text-4xl">{meta.icon}</span>
                <h3 className="font-semibold">{meta.label}</h3>
                <span className="text-xs text-slate-500">{count} Simulations</span>
              </div>
            );
          })}
        </div>
      </section>

      {/* Simulations Grid */}
      <section className="px-6 pb-24 max-w-7xl mx-auto">
        <h2 className="text-2xl font-bold mb-8 flex items-center gap-3">
          <span className="w-8 h-1 bg-blue-500 rounded-full"></span>
          Explore Simulations
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {SIMULATIONS.map((sim) => {
            const subjectMeta = SUBJECTS[sim.subject];
            const isImplemented = sim.implemented;
            
            const CardContent = () => (
              <div className={`h-full bg-slate-900 border ${isImplemented ? 'border-slate-700 hover:border-slate-500 hover:shadow-lg hover:shadow-blue-900/20' : 'border-slate-800 opacity-60'} rounded-2xl p-6 transition-all flex flex-col group`}>
                <div className="flex justify-between items-start mb-4">
                  <span className="text-4xl bg-slate-800 w-16 h-16 rounded-2xl flex items-center justify-center shadow-inner">
                    {sim.icon}
                  </span>
                  <div className="flex flex-col items-end gap-2">
                    {isImplemented ? (
                      <span className="px-2 py-1 text-[10px] font-bold uppercase tracking-wider bg-blue-500/10 text-blue-400 rounded-lg border border-blue-500/20">
                        Available
                      </span>
                    ) : (
                      <span className="px-2 py-1 text-[10px] font-bold uppercase tracking-wider bg-slate-800 text-slate-400 rounded-lg">
                        Coming Soon
                      </span>
                    )}
                    <span className={`px-2 py-1 text-[10px] font-bold uppercase tracking-wider rounded-lg border ${
                      sim.difficulty === 'easy' ? 'border-green-500/20 text-green-400' :
                      sim.difficulty === 'medium' ? 'border-amber-500/20 text-amber-400' :
                      'border-red-500/20 text-red-400'
                    }`}>
                      {sim.difficulty}
                    </span>
                  </div>
                </div>
                
                <h3 className={`text-xl font-bold mb-2 text-slate-100 ${isImplemented ? 'group-hover:text-blue-400 transition-colors' : ''}`}>
                  {sim.title}
                </h3>
                <p className="text-sm text-slate-400 flex-1 mb-6">
                  {sim.description}
                </p>
                
                <div className="mt-auto flex items-center justify-between text-sm font-medium text-slate-500">
                  <span className="flex items-center gap-2">
                    <span className="text-lg">{subjectMeta.icon}</span>
                    {subjectMeta.label}
                  </span>
                </div>
              </div>
            );

            return isImplemented ? (
              <Link key={sim.id} href={sim.path} className="outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded-2xl block h-full">
                <CardContent />
              </Link>
            ) : (
              <div key={sim.id} className="cursor-not-allowed h-full">
                <CardContent />
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
