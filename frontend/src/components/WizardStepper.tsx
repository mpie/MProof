'use client';

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck } from '@fortawesome/free-solid-svg-icons';

export type WizardStep = 1 | 2 | 3;

export interface WizardStepperStep {
  title: string;
  subtitle?: string;
}

export interface WizardStepperProps {
  activeStep: WizardStep;
  steps?: [WizardStepperStep, WizardStepperStep, WizardStepperStep];
}

const defaultSteps: [WizardStepperStep, WizardStepperStep, WizardStepperStep] = [
  { title: 'Selecteer subject', subtitle: 'Context + naam' },
  { title: 'Upload document', subtitle: 'Sleep of klik' },
  { title: 'Bekijk analyse', subtitle: 'Live status' },
];

function clampStep(step: number): WizardStep {
  if (step <= 1) return 1;
  if (step >= 3) return 3;
  return step as WizardStep;
}

function progressForStep(step: WizardStep): number {
  switch (step) {
    case 1: return 0;
    case 2: return 50;
    case 3: return 100;
  }
}

export function WizardStepper({ activeStep, steps = defaultSteps }: WizardStepperProps) {
  const safeStep = clampStep(activeStep);
  const progress = progressForStep(safeStep);

  return (
    <div className="w-full">
      {/* Progress bar */}
      <div className="relative h-1 bg-white/10 rounded-full overflow-hidden mb-4">
        <div
          className="absolute left-0 top-0 h-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Steps */}
      <div className="flex justify-between">
        {steps.map((step, index) => {
          const stepNum = (index + 1) as WizardStep;
          const isActive = safeStep === stepNum;
          const isCompleted = safeStep > stepNum;

          return (
            <div key={index} className="flex flex-col items-center flex-1">
              <div
                className={`
                  w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold
                  transition-all duration-300
                  ${isCompleted
                    ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white'
                    : isActive
                      ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white'
                      : 'bg-white/10 text-white/40'
                  }
                `}
              >
                {isCompleted ? (
                  <FontAwesomeIcon icon={faCheck} className="w-4 h-4" />
                ) : (
                  stepNum
                )}
              </div>
              <div className="mt-2 text-center">
                <div
                  className={`text-xs font-medium transition-colors duration-300 ${
                    isActive || isCompleted ? 'text-white' : 'text-white/40'
                  }`}
                >
                  {step.title}
                </div>
                {step.subtitle && (
                  <div className="text-[10px] text-white/30 mt-0.5 hidden sm:block">
                    {step.subtitle}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
